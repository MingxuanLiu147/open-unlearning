<template>
  <div class="cmd-section">
    <div class="cmd-header">
      <span class="section-title">{{ $t('workshop.command') }}</span>
      <el-button text size="small" @click="copy">{{ $t('workshop.copyCmd') }}</el-button>
    </div>
    <pre class="code-block">{{ commandText }}</pre>
    <div class="run-actions">
      <el-button
        type="primary"
        :loading="runnerStore.running"
        @click="startRun"
        :disabled="!canRun"
      >
        {{ $t('workshop.startRun') }}
      </el-button>
      <el-button
        type="danger"
        v-if="runnerStore.running"
        @click="stopRun"
      >
        {{ $t('workshop.stopRun') }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useExperimentStore } from '@/stores/experiment'
import { useRunnerStore } from '@/stores/runner'
import { runnerApi } from '@/api'
import { useRouter } from 'vue-router'

const store = useExperimentStore()
const runnerStore = useRunnerStore()
const router = useRouter()

const canRun = computed(() => store.selectedModel && store.selectedTrainer)

const commandText = computed(() => {
  const parts = ['python src/train.py']
  parts.push(`  --config-name=${store.mode}.yaml`)
  if (store.selectedExperiment) parts.push(`  experiment=${store.selectedExperiment}`)
  if (store.selectedModel) parts.push(`  model=${store.selectedModel}`)
  if (store.selectedTrainer) parts.push(`  trainer=${store.selectedTrainer}`)
  parts.push(`  task_name=${store.taskName}`)
  if (store.selectedEval) parts.push(`  eval=${store.selectedEval}`)

  const ds = store.selectedDatasets
  if (ds.forget) parts.push(`  data/datasets@data.forget=${ds.forget}`)
  if (ds.retain) parts.push(`  data/datasets@data.retain=${ds.retain}`)
  if (ds.edit) parts.push(`  data/datasets@data.edit=${ds.edit}`)
  if (ds.train) parts.push(`  data/datasets@data.train=${ds.train}`)

  parts.push(`  trainer.args.seed=${store.seed}`)
  return parts.join(' \\\n')
})

function copy() {
  navigator.clipboard.writeText(commandText.value).then(() => {
    ElMessage.success('Copied!')
  })
}

async function startRun() {
  runnerStore.clearLogs()
  const body = {
    mode: store.mode,
    model: store.selectedModel,
    trainer: store.selectedTrainer,
    experiment: store.selectedExperiment || undefined,
    task_name: store.taskName,
    overrides: store.overrides,
    eval_suite: store.selectedEval || undefined,
    gpu: store.gpu,
  }
  const res = await runnerApi.start(body)
  if (res.ok) {
    runnerStore.running = true
    runnerStore.command = res.command
    router.push('/monitor')
  } else {
    ElMessage.error(res.error || 'Failed to start')
  }
}

async function stopRun() {
  await runnerApi.stop()
  runnerStore.running = false
}
</script>

<style scoped>
.cmd-section { margin-top: 12px; }
.cmd-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.run-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
</style>
