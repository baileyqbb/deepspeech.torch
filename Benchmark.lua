-- Use this script to measure how long it takes to do a forward/backward pass through the standard deepspeech model.
require 'UtilsMultiGPU'

require 'cudnn'
cudnn.benchmark = true

-- Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-batchSize', 20, 'Batch size in training')
cmd:option('-LSTM', false, 'Use LSTMs rather than RNNs')
cmd:option('-hiddenSize', 1760, 'RNN hidden sizes')
cmd:option('-nbOfHiddenLayers', 7, 'Number of rnn layers')
cmd:option('-seqLength', 1500, 'Size of sequence to benchmark')

local opt = cmd:parse(arg)
local input = torch.Tensor(opt.batchSize, 1, 161, opt.seqLength)
if opt.nGPU > 0 then
  require 'cudnn'
  require 'cunn'
  input = input:cuda()
end

local model = require(opt.modelName)
model = model[1](opt)
model = makeDataParallel(model, opt.nGPU)

local time_fwd = 0
local time_bwd = 0
local loops = 100
local loops_burnin = 50

for i = 1, loops do

  local timer = torch.Timer()

  timer:reset()

  local output = model:forward(input)
  local time_fwd_i = timer:time().real

  timer:reset()

  model:backward(input, output)

  local time_bwd_i = timer:time().real

  if i >= loops_burnin then
    time_fwd = time_fwd + time_fwd_i / (loops - loops_burnin)
    time_bwd = time_bwd + time_bwd_i / (loops - loops_burnin)
  end
end

print(("Forward Time: %.3fs Backward Time: %.3fs Full Time: %.3fs"):format(time_fwd, time_bwd, time_fwd + time_bwd))