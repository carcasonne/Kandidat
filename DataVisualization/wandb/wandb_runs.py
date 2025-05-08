from dataclasses import dataclass

@dataclass(frozen=True)
class ModelID:
    id: str
    display_name: str


AST_2K = ModelID("lw2r1isa", "AST (2K)")
AST_20K = ModelID("vog5pazp", "AST (20K)")
AST_100K = ModelID("gshghk2u", "AST (100K)")

PRETRAINED_2K = ModelID("bkwc7ac6", "Pretrained (2K)")
PRETRAINED_20K = ModelID("kzanqk87", "Pretrained (20K)")
PRETRAINED_100K = ModelID("sn68v90l", "Pretrained (100K)")
