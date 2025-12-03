
modelName = 'speechTransformerCNN_LabelSmooth0.1_withoutLayerNorm'

args = {}
args['outputDir'] = '/home/chenggong/UCLA-ECE243-FALL2025/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/chenggong/UCLA-ECE243-FALL2025/data/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0008 
args['lrEnd'] = 0.00008

args['warmupSteps'] = 500

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
args['optimizer'] = 'adamw'

args['lambda_cr'] = 0.0  # disable CR-CTC here (set >0 to enable)

# --- Transformer Specifics ---
args['use_transformer'] = True
args['nhead'] = 6           
args['dim_feedforward'] = 1536 
args['timeMasking'] = True
args['featureMasking'] = True # Enable Feature Masking
args['use_rope'] = True

args['labelSmoothing'] = 0.0  # Enable Label Smoothing with 0.1 factor

args['earlyStoppingPatience'] = 50  # Early stopping patience
args['use_layer_norm'] = True  # Enable LayerNorm in encoder/decoder
args['gradClip'] = 5.0  # Gradient clipping threshold
args['use_rope'] = True  # Enable RoPE in transformer attention

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
