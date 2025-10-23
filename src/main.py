from subprocess import Popen
import os,shlex

GPU_NUMBER                        = 0
SWEEP_ID                          = 'd7jrytds'
WANDB_USERNAME                    = 'dsingh93'
WANDB_KEY                         = '558a28f84276e6957673c5ad3656ea9b82cbd29d' 
SEED                              = -1
WB_ENTITY_NAME                    = 'uitmlg'
WB_PROJECT_NAME                   = 'ssl'

if os.environ.get('SINGULARITY_NAME'):
  WANDB_EXEC_LOC= os.path.join(os.sep,'users',os.path.expanduser('~'),'.local','bin','wandb')
else:
  WANDB_EXEC_LOC='wandb'


#cmd_sweep ="""{} agent uitmlg/ssl/{}""".format(WANDB_EXEC_LOC,SWEEP_ID)



os.environ['WANDB_API_KEY']       = WANDB_KEY
os.environ['WANDB_USERNAME']      = WANDB_USERNAME
os.environ['GPU']                 = str(GPU_NUMBER)
os.environ['SEED']                = str(SEED)
os.environ['WB_ENTITY_NAME']      = str(WB_ENTITY_NAME)
os.environ['WB_PROJECT_NAME']     = str(WB_PROJECT_NAME)

print('from the main function in supcm')
exit()

cmd_list =[ 
           """python train.py --config='stl10_NONE_ce_600_config' --load_args=False --logmode=f""",
          ]

#cmd_list =[ """python build_dataset.py --seed 42 --dataset all""" ]
procs = [ Popen(shlex.split(cmd),shell=False) for cmd in cmd_list]
for p in procs:
  p.wait()