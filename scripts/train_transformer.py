
modelName = 'speechTransformerCNN_SpecAug_LR0008'

args = {}
args['outputDir'] = '/home/tianlezheng/UCLA-ECE243-FALL2025/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/tianlezheng/UCLA-ECE243-FALL2025/data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0008
args['lrEnd'] = 0.00008
args['nUnits'] = 384        
args['nBatch'] = 20000      
args['nLayers'] = 8

args['scheduler_type'] = 'cosine_warmup'  # 'linear', 'cosine', 'cosine_warmup'
args['warmup_steps'] = 800
args['early_stopping_patience'] = 50
args['max_grad_norm'] = 1.0
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

args['use_transformer'] = True
args['nhead'] = 6           
args['dim_feedforward'] = 1536 
args['timeMasking'] = True
args['featureMasking'] = True

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
