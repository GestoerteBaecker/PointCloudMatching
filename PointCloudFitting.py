import copy
import numpy as np
import os


class PointCloudFitting:
    def __init__(self, source_points, dest_points, correspondencies):
        if len(correspondencies) < 3:
            raise Exception("There has to be at least 3 matched points for fitting")
        self.source_vec = np.array([])
        self.dest_vec = np.array([])
        self.source_points = source_points
        self.dest_points = dest_points
        self.PrepareInputData(source_points, dest_points)

        self.fitting_equations = []
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)
        self.X = np.zeros(6)

        self.nearing = self.ComputeNearing()

        for id in self.source_points.keys():
            x_s, y_s, z_s = self.source_points[id]

            x_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: X_0 + (np.cos(kappa)*np.cos(phi)*x + (-np.sin(kappa)*np.cos(phi))*y + np.sin(phi)*z)
            y_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: Y_0 + ((np.sin(kappa)*np.cos(omega) + np.sin(omega)*np.sin(phi)*np.cos(kappa))*x + (-np.sin(kappa)*np.sin(omega)*np.sin(phi) + np.cos(kappa)*np.cos(omega))*y + (-np.sin(omega)*np.cos(phi))*z)
            z_eq = lambda omega, phi, kappa, X_0, Y_0, Z_0, x=x_s, y=y_s, z=z_s: Z_0 + ((np.sin(kappa)*np.sin(omega) - np.sin(phi)*np.cos(kappa)*np.cos(omega))*x + (np.sin(kappa)*np.sin(phi)*np.cos(omega) + np.sin(omega)*np.cos(kappa))*y + (np.cos(omega)*np.cos(phi))*z)

            eq = [x_eq, y_eq, z_eq]
            self.fitting_equations = self.fitting_equations + eq

        self.StartAdjustment()

    def PrepareInputData(self, source_points, dest_points):
        for i_source in source_points.keys():
            if not i_source in dest_points.keys():
                raise Exception("Es wurden tlw. nicht identische Punkt-IDs übergeben")
            self.source_vec = np.hstack((self.source_vec, source_points[i_source]))
            self.dest_vec = np.hstack((self.dest_vec, dest_points[i_source]))


    def MakeJacobian(self):
        matrix = np.zeros((len(self.fitting_equations), 6))
        for i in range(len(self.fitting_equations)):
            for j in range(6):
                matrix[i, j] = Differentiate(self.fitting_equations[i], self.nearing, j)
        return matrix


    def ComputeNearing(self, index_adding=0):
        # Puntke finden
        ids = list(self.source_points.keys())
        if 2+index_adding >=len(ids):
            raise Exception("Es wurden nicht genügend oder flächig verteilte Punkte angegeben")
        # Source
        p1 = self.source_points[ids[0]]
        p2 = self.source_points[ids[1]]
        p3 = self.source_points[ids[2+index_adding]]

        # Ziel
        P1 = self.dest_points[ids[0]]
        P2 = self.dest_points[ids[1]]
        P3 = self.dest_points[ids[2 + index_adding]]

        # Bildes eines Hilfssystems: Es wird von den drei identischen Punkten aufgespannt; es werden Trans-matrizen aufgestellt, die von beiden Systemen da rein transformieren, dann kann der Übergang vom X über Hilfsystem zu x stattfinden
        p12 = p2 - p1
        p13 = p3 - p1

        P12 = P2 - P1
        P13 = P3 - P1

        u = p12/np.linalg.norm(p12)
        w = np.cross(u, p13)
        if np.linalg.norm(w) < 1e-5:
            self.ComputeNearing(index_adding+1)
        w = w/np.linalg.norm(w)
        v = np.cross(w, u)

        U = P12/np.linalg.norm(P12)
        W = np.cross(U, P13)
        if np.linalg.norm(W) < 1e-5:
            self.ComputeNearing(index_adding+1)
        W = W/np.linalg.norm(W)
        V = np.cross(W, U)

        # Transformationsmatrizen des Hilfssystems in die jeweiligen Systeme
        R_U_X = MakeBasis(U, V, W)
        r_u_x = MakeBasis(u, v, w)
        R = R_U_X.dot(r_u_x.T)

        # Differentiate der Winkel
        phi = np.arcsin(R[0, 2])
        omega = np.arctan(-R[1, 2] / R[2, 2])
        kappa = np.arctan(-R[0, 1] / R[0, 0])

        # Maßstab
        m = float(((np.linalg.norm(P12) / np.linalg.norm(p12)) + (np.linalg.norm(P13) / np.linalg.norm(p13)) + (np.linalg.norm(P2 - P3)/np.linalg.norm(p2 - p3)))/3)
        if m < 0.95 or m > 1.05:
            raise Exception("Es liegt eine unmaßstäbliche Transformation vor")

        # Linearer Anteil der Verschiebung, daher alle auf 0
        X_S = 1/3 * (P1 + P2 + P3)
        x_s = 1/3 * (p1 + p2 + p3)
        X_0, Y_0, Z_0 = X_S - R.dot(x_s)

        return [omega, phi, kappa, float(X_0), float(Y_0), float(Z_0)]


    def Adjust(self):
        obs_approx = self.dest_vec * 0
        for i in range(len(self.dest_vec)):
            obs_approx[i] = self.fitting_equations[i](*self.nearing)
        obs_abridged = self.dest_vec - obs_approx
        A = self.MakeJacobian()
        n = (A.T).dot(obs_abridged)
        N = (A.T).dot(A)
        try:
            Q = np.linalg.inv(N)
        except:
            raise Exception("Ausgleichung nicht möglich")
        x_abridged = Q.dot(n)
        self.X = self.nearing + x_abridged


    def StartAdjustment(self):
        accuracys = [10**-8, 10**-8, 10**-8, 10**-5, 10**-5, 10**-5]
        for i in range(1000):
            #mögliche neuberechnung, falls unterschied zwischen näherungswerten und ausgeglichenen werten eine grenze (1/1000) nicht unterschreitet
            converged = True
            for j in range(len(self.X)):
                converged = converged and (abs(self.nearing[j] - self.X[j])) < accuracys[j]

            if not converged:
                self.nearing = list(self.X)
                self.Adjust()
            else:
                self.translation = self.X[3:6]
                self.rotation = MakeTransformationMatrix(self.X[0], self.X[1], self.X[2], 1)
                return
        raise Exception("Die Ausgleichung ist nicht konvergiert")

    def Transform(self, point):
        return self.translation + self.rotation.dot(point)

    def GetRotation(self):
        return self.rotation

    def GetTranslation(self):
        return self.translation

# csv-Datei: erster Wert ID; folgende 3 Koordinaten. Trennzeichen ;
def ReadPointsFromFile(filename):
    if not os.path.exists(filename):
        raise Exception("Dateiname existiert nicht")
    points = dict()
    ex = Exception("Die Daten liegen nicht durch ';' getrennt vor oder können nicht zu Zahlen konvertiert werden")
    with open(filename, "r") as file:
        for row in file.readlines():
            data = row.split(";")
            try:
                points[int(data[0])] = np.array([float(entry) for entry in data[1:4]])
            except:
                raise ex
    return points

def main():
    try:
        source_points = ReadPointsFromFile(input("Bitte geben Sie den Dateinamen der Punkte des Quellsystems an: "))
        dest_points = ReadPointsFromFile(input("Bitte geben Sie den Dateinamen der Punkte des Zielsystems an: "))

        adjustment = Adjustment(source_points, dest_points)
        rotation = adjustment.GetRotation()
        translation = adjustment.GetTranslation()

        phi = np.arcsin(rotation[0, 2]) * 180 / np.pi
        omega = np.arctan(-rotation[1, 2] / rotation[2, 2]) * 180 / np.pi
        kappa = np.arctan(-rotation[0, 1] / rotation[0, 0]) * 180 / np.pi

        print("--- Ausgabe ---")
        print("Translation:", translation)
        print("Rotation:", rotation)
        print("Rotationwinkel:", )
        print("    Omega:", omega)
        print("    Phi:", phi)
        print("    Kappa:", kappa)
        print("------------------------------------")

        input()
    except Exception as ex:
        print(ex)

#########################################

def Test():
    # TEST:
    p1 = np.array([10,5,2])
    p2 = np.array([0,-8,4])
    p3 = np.array([-9,-6,7])
    p4 = np.array([-11,4,4])
    p5 = np.array([3,-6,8])

    omega = 30
    phi = 10
    kappa = 45
    omega *= np.pi / 180
    phi *= np.pi / 180
    kappa *= np.pi / 180
    rotation = MakeTransformationMatrix(omega, phi, kappa, 1)
    translation = np.array([10,5,7])

    P1 = translation + rotation.dot(p1) + (np.random.rand(3)-0.5)/10
    P2 = translation + rotation.dot(p2) + (np.random.rand(3)-0.5)/10
    P3 = translation + rotation.dot(p3) + (np.random.rand(3)-0.5)/10
    P4 = translation + rotation.dot(p4) + (np.random.rand(3)-0.5)/10
    P5 = translation + rotation.dot(p5) + (np.random.rand(3)-0.5)/10

    quelle = {0: p1, 1: p2, 2: p3, 3: p4, 4: p5}
    ziel = {0: P1, 1: P2, 2: P3, 3: P4, 4: P5}

    adjustment = Adjustment(quelle, ziel)
    rotation_new = adjustment.GetRotation()
    translation_new = adjustment.GetTranslation()

    phi_new = np.arcsin(rotation_new[0,2]) *180/np.pi
    omega_new = np.arctan(-rotation_new[1,2] / rotation_new[2,2]) *180/np.pi
    kappa_new = np.arctan(-rotation_new[0,1] / rotation_new[0,0]) *180/np.pi

    P1_new = adjustment.Transform(p1)
    P2_new = adjustment.Transform(p2)
    P3_new = adjustment.Transform(p3)
    P4_new = adjustment.Transform(p4)
    P5_new = adjustment.Transform(p5)
    i=0

if __name__ == "__main__":
    main()