import sys
sys.path.append("./")

from configure import elementVDWRadius,elementVDWWellDepth

class Mol2Atom(object):
    """description of class"""

    def __init__(self,config, atom_id, atom_name, x, y, z, atom_type, subst_id=None, subst_name=None, charge=None,
                 status_bit=None, auto4_atom_type=None, auto4_partial_charge=0):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.atom_type = atom_type
        self.subst_id = subst_id
        self.subst_name = subst_name
        self.charge = charge
        self.status_bit = status_bit
        self.auto4_atom_type = auto4_atom_type
        self.auto4_partial_charge = None
        if auto4_partial_charge != None:
            self.auto4_partial_charge = auto4_partial_charge
        self.vdw_welldepth = elementVDWWellDepth(self.atom_type)
        self.vdw_radius = elementVDWRadius(self.atom_type, 0)


class PdbqtAtom():
    def __init__(self, atom_id, x, y, z, atom_type, partial_charge, vdw=None, elec=None, atom_name=None,
                 subst_name=None, atom_status=None):
        self.atom_id = atom_id
        self.atom_name = atom_name
        self.subst_name = subst_name
        self.atom_status = atom_status
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vdw = vdw
        self.elec = elec
        self.partial_charge = float(partial_charge)
        self.atom_type = atom_type