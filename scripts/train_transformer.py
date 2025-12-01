
modelName = 'speechTransformerCNN_ROPE'

args = {}
args['outputDir'] = '/home/alex/Downloads/UCLA-ECE243-FALL2025/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/alex/Downloads/UCLA-ECE243-FALL2025/data/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0005 
args['lrEnd'] = 0.00005
args['nUnits'] = 384        
args['nBatch'] = 12000      
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
args['optimizer'] = 'adamw'

args['lambda_cr'] = 0.0  # disable CR-CTC here (set >0 to enable)

# --- Transformer Specifics ---
args['use_transformer'] = True
args['nhead'] = 6           
args['dim_feedforward'] = 1536 
args['timeMasking'] = True
args['featureMasking'] = True # Enable Feature Masking
args['use_rope'] = True

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
