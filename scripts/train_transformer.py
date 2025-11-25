
modelName = 'speechTransformerCNN_SpecAug'

args = {}
args['outputDir'] = '/home/harry/projects/ucla/UCLA-ECE243-FALL2025/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/harry/projects/ucla/UCLA-ECE243-FALL2025/data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0005 
args['lrEnd'] = 0.00005
args['nUnits'] = 384        
args['nBatch'] = 16000      
args['nLayers'] = 8         
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.3       
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 0
args['bidirectional'] = False 
args['l2_decay'] = 0.01

# --- Transformer Specifics ---
args['use_transformer'] = True
args['nhead'] = 6           
args['dim_feedforward'] = 1536 
args['timeMasking'] = True
args['featureMasking'] = True # Enable Feature Masking

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
