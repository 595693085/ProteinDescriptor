class Protein:
    def __init__(self):
        # the atoms and substructures of the protein,rank from index 1
        self.PDB_ID = None
        self.Mol2AtomsList = []
        self.Mol2AtomsList.append(None)
        self.Mol2SubstructuresList = []
        self.Mol2SubstructuresList.append(None)

        self.Mol2MinCoordinate = [float('inf'), float('inf'), float('inf')]
        self.Mol2MaxCoordinate = [float('-inf'), float('-inf'), float('-inf')]
        self.Mol2MinCoorNp = None
        self.Mol2MaxCoorNp = None
        self.Mol2CoorSpanNp = None
        self.X_Length = 0
        self.Y_length = 0
        self.Z_length = 0

        self.PdbqtAtomsList = []
        self.PdbqtAtomsList.append(None)
        self.PdbqtSubstructuresList = []
        self.PdbqtSubstructuresList.append(None)

        self.PdbqtMinCoordinate = [float('inf'), float('inf'), float('inf')]
        self.PdbqtMaxCoordinate = [float('-inf'), float('-inf'), float('-inf')]

        self.SiteCentralCoorX = 0
        self.SiteCentralCoorY = 0
        self.SiteCentralCoorZ = 0
        self.SiteVolume = 0

        # the grids of the protein
        self.GridChannels = []

    def AddMol2Atom(self, mol2_atom):
        self.Mol2AtomsList.append(mol2_atom)

        self.Mol2MinCoordinate[0]=min(mol2_atom.x,self.Mol2MinCoordinate[0])
        self.Mol2MinCoordinate[1]=min(mol2_atom.y,self.Mol2MinCoordinate[1])
        self.Mol2MinCoordinate[2]=min(mol2_atom.z,self.Mol2MinCoordinate[2])
        self.Mol2MaxCoordinate[0]=max(mol2_atom.x,self.Mol2MaxCoordinate[0])
        self.Mol2MaxCoordinate[1]=max(mol2_atom.y,self.Mol2MaxCoordinate[1])
        self.Mol2MaxCoordinate[2]=max(mol2_atom.z,self.Mol2MaxCoordinate[2])

        self.X_Length = self.Mol2MaxCoordinate[0] - self.Mol2MinCoordinate[0]
        self.Y_length = self.Mol2MaxCoordinate[1] - self.Mol2MinCoordinate[1]
        self.Z_length = self.Mol2MaxCoordinate[2] - self.Mol2MinCoordinate[2]
        # print("add atom over")

    def AddMol2Residue(self, mol2_substructure):
        self.Mol2SubstructuresList.append(mol2_substructure)

    def AddPdbqtAtom(self, pdbqt_atom):
        self.PdbqtAtomsList.append(pdbqt_atom)
        # print("add atom over")

    def AddChannel(self, grid):
        self.GridChannels.append(grid)