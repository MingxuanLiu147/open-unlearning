import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useExperimentStore = defineStore('experiment', () => {
  const mode = ref<'unlearn' | 'inject' | 'edit' | 'eval'>('unlearn')
  const selectedModel = ref('')
  const selectedTrainer = ref('')
  const selectedDatasets = ref<Record<string, string>>({})
  const selectedEval = ref('')
  const selectedExperiment = ref('')
  const taskName = ref('my_experiment')
  const gpu = ref('0')
  const seed = ref(42)
  const params = ref<Record<string, any>>({
    learning_rate: 1e-5,
    num_epochs: 5,
    batch_size: 4,
    gradient_accumulation_steps: 1,
    max_length: 512,
    warmup_ratio: 0.1,
  })

  const overrides = computed(() => {
    const o: Record<string, any> = {}
    if (params.value.learning_rate) o['trainer.args.learning_rate'] = params.value.learning_rate
    if (params.value.num_epochs) o['trainer.args.num_train_epochs'] = params.value.num_epochs
    if (params.value.batch_size) o['trainer.args.per_device_train_batch_size'] = params.value.batch_size
    if (params.value.gradient_accumulation_steps) o['trainer.args.gradient_accumulation_steps'] = params.value.gradient_accumulation_steps
    if (params.value.max_length) o['data.max_length'] = params.value.max_length
    if (params.value.warmup_ratio) o['trainer.args.warmup_ratio'] = params.value.warmup_ratio
    if (seed.value) o['trainer.args.seed'] = seed.value
    return o
  })

  function applyConfig(cfg: Record<string, any>) {
    if (cfg.mode) mode.value = cfg.mode
    if (cfg.model) selectedModel.value = cfg.model
    if (cfg.trainer) selectedTrainer.value = cfg.trainer
    if (cfg.datasets) selectedDatasets.value = cfg.datasets
    if (cfg.eval) selectedEval.value = cfg.eval
    if (cfg.params) Object.assign(params.value, cfg.params)
  }

  function reset() {
    selectedModel.value = ''
    selectedTrainer.value = ''
    selectedDatasets.value = {}
    selectedEval.value = ''
    selectedExperiment.value = ''
  }

  return {
    mode, selectedModel, selectedTrainer, selectedDatasets,
    selectedEval, selectedExperiment, taskName, gpu, seed,
    params, overrides, applyConfig, reset,
  }
})
