import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface ModelInfo {
  name: string
  modality?: string
  path?: string
  dtype?: string
  attn?: string
}

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
  const modelPath = ref('')
  const tokenizerPath = ref('')
  const outputDir = ref('')
  const modelCatalog = ref<ModelInfo[]>([])
  const datasetListVersion = ref(0)
  const params = ref<Record<string, any>>({
    learning_rate: 1e-5,
    num_epochs: 5,
    batch_size: 4,
    gradient_accumulation_steps: 1,
    max_length: 512,
    warmup_ratio: 0.1,
  })

  function setModelCatalog(models: ModelInfo[]) {
    modelCatalog.value = models
  }

  const selectedModality = computed(() => {
    const m = modelCatalog.value.find((x) => x.name === selectedModel.value)
    return m?.modality === 'multimodal' ? 'multimodal' : 'text'
  })

  const pathOverrides = computed(() => {
    const o: Record<string, string> = {}
    if (modelPath.value.trim()) {
      o['model.model_args.pretrained_model_name_or_path'] = modelPath.value.trim()
    }
    const tok = tokenizerPath.value.trim() || modelPath.value.trim()
    if (tok) {
      o['model.tokenizer_args.pretrained_model_name_or_path'] = tok
    }
    if (outputDir.value.trim()) {
      o['paths.output_dir'] = outputDir.value.trim()
    }
    return o
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
    Object.assign(o, pathOverrides.value)
    return o
  })

  function bumpDatasetList() {
    datasetListVersion.value += 1
  }

  function applyConfig(cfg: Record<string, any>) {
    if (cfg.mode && typeof cfg.mode === 'string') mode.value = cfg.mode as any

    const modelVal = cfg.model
    if (modelVal) {
      if (typeof modelVal === 'string') {
        selectedModel.value = modelVal
      } else if (typeof modelVal === 'object' && modelVal.base_model) {
        selectedModel.value = modelVal.base_model
      }
    }

    const trainerVal = cfg.trainer
    if (trainerVal) {
      if (typeof trainerVal === 'string') {
        selectedTrainer.value = trainerVal
      } else if (typeof trainerVal === 'object' && trainerVal.name) {
        selectedTrainer.value = trainerVal.name
      }
    }

    if (cfg.datasets) {
      if (typeof cfg.datasets === 'object') {
        const ds: Record<string, string> = {}
        for (const [k, v] of Object.entries(cfg.datasets)) {
          ds[k] = typeof v === 'string' ? v : String(v)
        }
        selectedDatasets.value = ds
      }
    }

    if (cfg.eval) {
      selectedEval.value = typeof cfg.eval === 'string' ? cfg.eval : String(cfg.eval)
    }

    if (cfg.params && typeof cfg.params === 'object') {
      Object.assign(params.value, cfg.params)
    }

    if (typeof cfg.model_path === 'string') modelPath.value = cfg.model_path
    if (typeof cfg.output_dir === 'string') outputDir.value = cfg.output_dir
  }

  function reset() {
    selectedModel.value = ''
    selectedTrainer.value = ''
    selectedDatasets.value = {}
    selectedEval.value = ''
    selectedExperiment.value = ''
    modelPath.value = ''
    tokenizerPath.value = ''
    outputDir.value = ''
  }

  return {
    mode,
    selectedModel,
    selectedTrainer,
    selectedDatasets,
    selectedEval,
    selectedExperiment,
    taskName,
    gpu,
    seed,
    modelPath,
    tokenizerPath,
    outputDir,
    modelCatalog,
    datasetListVersion,
    params,
    overrides,
    pathOverrides,
    selectedModality,
    setModelCatalog,
    bumpDatasetList,
    applyConfig,
    reset,
  }
})
