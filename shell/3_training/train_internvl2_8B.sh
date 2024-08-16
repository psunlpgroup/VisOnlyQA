source $CONDA_SH
conda activate internvl

TRAIN_DATA=3d__size sh shell/3_training/__train_internvl2_8B.sh
TRAIN_DATA=3d__angle sh shell/3_training/__train_internvl2_8B.sh

TRAIN_DATA=syntheticgeometry__triangle sh shell/3_training/__train_internvl2_8B.sh
TRAIN_DATA=syntheticgeometry__quadrilateral sh shell/3_training/__train_internvl2_8B.sh

TRAIN_DATA=syntheticgeometry__angle sh shell/3_training/__train_internvl2_8B.sh
TRAIN_DATA=syntheticgeometry__area sh shell/3_training/__train_internvl2_8B.sh

TRAIN_DATA=syntheticgeometry__length sh shell/3_training/__train_internvl2_8B.sh

conda deactivate
