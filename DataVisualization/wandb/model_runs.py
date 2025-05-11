from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from enum import Enum

class ProjectType(Enum):
    """Types of projects in W&B"""
    AST = "Kandidat-AST"
    PRETRAINED = "Kandidat-Pre-trained"
    
    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class ModelRun:
    """Information about a specific model run"""
    id: str
    display_name: str
    shortname: str
    project: ProjectType
    tags: List[str] = None
    description: Optional[str] = None
    
    @property
    def full_path(self) -> str:
        """Get the full path to the run (entity/project/id)"""
        return f"Holdet_thesis/{self.project}/{self.id}"
    
    def __str__(self) -> str:
        return self.display_name

# Define run groups
class DataSize(Enum):
    SMALL = "2K"
    MEDIUM = "20K"
    LARGE = "100K"
    
    
# AST Model Runs
AST_2K = ModelRun(id="lw2r1isa", display_name="AST (2K)", shortname="AST2K", project=ProjectType.AST, 
                  tags=["AST", "2K", "small"])
AST_20K = ModelRun(id="vog5pazp", display_name="AST (20K)", shortname="AST20K", project=ProjectType.AST,
                   tags=["AST", "20K", "medium"])
AST_100K = ModelRun(id="gshghk2u", display_name="AST (100K)", shortname="AST100K", project=ProjectType.AST,
                    tags=["AST", "100K", "large"])

# Pretrained Model Runs
PRETRAINED_2K = ModelRun(id="bkwc7ac6", display_name="Pretrained (2K)", shortname="PRETRAINED2K",  
                         project=ProjectType.PRETRAINED, tags=["pretrained", "2K", "small"])
PRETRAINED_20K = ModelRun(id="kzanqk87", display_name="Pretrained (20K)", shortname="PRETRAINED20K",
                          project=ProjectType.PRETRAINED, tags=["pretrained", "20K", "medium"])
PRETRAINED_100K = ModelRun(id="sn68v90l", display_name="Pretrained (100K)", shortname="PRETRAINED100K",
                           project=ProjectType.PRETRAINED, tags=["pretrained", "100K", "large"])

# Run collections for convenience
ALL_RUNS = [AST_2K, AST_20K, AST_100K, PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K]
AST_RUNS = [AST_2K, AST_20K, AST_100K]
PRETRAINED_RUNS = [PRETRAINED_2K, PRETRAINED_20K, PRETRAINED_100K]
SMALL_RUNS = [AST_2K, PRETRAINED_2K]
MEDIUM_RUNS = [AST_20K, PRETRAINED_20K]
LARGE_RUNS = [AST_100K, PRETRAINED_100K]

# Helper functions
def get_runs_by_tag(tag: str) -> List[ModelRun]:
    """Get all runs with a specific tag"""
    return [run for run in ALL_RUNS if run.tags and tag in run.tags]

def get_runs_by_size(size: DataSize) -> List[ModelRun]:
    """Get all runs with a specific data size"""
    return get_runs_by_tag(size.value)

def get_runs_by_model_type(model_type: str) -> List[ModelRun]:
    """Get all runs with a specific model type (AST or Pretrained)"""
    return get_runs_by_tag(model_type.lower())

def get_run_by_id(run_id: str) -> ModelRun:
    """Get a run by its ID"""
    for run in ALL_RUNS:
        if run.id == run_id:
            return run
    raise ValueError(f"No run found with ID '{run_id}'")