from ._base import Vanilla
from .FGD import FGD
from .PKD import PKD
from .FGD_PKD import FGD_PKD
from .DKD import DKD
from .Det_KD import Det_KD
from .det_reg_KD import det_reg_KD
from .reid_KD import reid_KD
# from .GroupGraph import GroupGraph
from .GraphRelation import GraphRelation
from .FGD_Relation import FGD_Relation
from .FGD_Det import FGD_Det
from .FGD_DKD import FGD_DKD
from .FGD_Det_Relation import FGD_Det_Relation
from .FGD_DKD_Relation import FGD_DKD_Relation
from .FGD_Det_Relation_ReID import FGD_Det_Relation_ReID
from .PKD_DKD_Relation import PKD_DKD_Relation
from .Det_Relation import Det_Relation
from .DKD_Relation import DKD_Relation
from .ReviewKD import ReviewKD

distiller_dict = {
    "NONE": Vanilla,
    "FGD": FGD,
    "PKD": PKD,
    "FGD_PKD": FGD_PKD,
    "DKD": DKD,
    "Det": Det_KD,
    "reg_KD": det_reg_KD,
    "reid_KD": reid_KD,
    # "GroupGraph": GroupGraph,
    "Relation": GraphRelation,
    "FGD_Relation": FGD_Relation,
    "FGD_Det_Relation": FGD_Det_Relation,
    "FGD_DKD_Relation": FGD_DKD_Relation,
    "FGD_Det_Relation_ReID": FGD_Det_Relation_ReID,
    "FGD_Det": FGD_Det,
    "FGD_DKD": FGD_DKD,
    "Det_Relation": Det_Relation,
    "DKD_Relation": DKD_Relation,
    "ReviewKD": ReviewKD,
    "PKD_DKD_Relation": PKD_DKD_Relation,
}