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
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { useExperimentStore } from '@/stores/experiment'
import { useRunnerStore } from '@/stores/runner'
import { runnerApi } from '@/api'
import { useRouter } from 'vue-router'

const { t } = useI18n()
const store = useExperimentStore()
const runnerStore = useRunnerStore()
const router = useRouter()

const canRun = computed(() => {
  if (store.mode === 'eval') {
    return !!(store.selectedModel && store.selectedEval)
  }
  return !!(store.selectedModel && store.selectedTrainer)
})

function safeStr(v: any): string {
  if (typeof v === 'string') return v
  if (v == null) return ''
  return String(v)
}

const commandText = computed(() => {
  if (store.mode === 'eval') {
    const parts = ['python src/eval.py', '  --config-name=eval.yaml']
    if (store.selectedEval) parts.push(`  eval=${safeStr(store.selectedEval)}`)
    if (store.selectedModel) parts.push(`  model=${safeStr(store.selectedModel)}`)
    parts.push(`  task_name=${store.taskName}`)
    for (const [k, v] of Object.entries(store.overrides)) {
      parts.push(`  ${k}=${safeStr(v)}`)
    }
    return parts.join(' \\\n')
  }

  const parts = ['python src/train.py']
  parts.push(`  --config-name=${store.mode}.yaml`)
  if (store.selectedExperiment) parts.push(`  experiment=${safeStr(store.selectedExperiment)}`)
  if (store.selectedModel) parts.push(`  model=${safeStr(store.selectedModel)}`)
  if (store.selectedTrainer) parts.push(`  trainer=${safeStr(store.selectedTrainer)}`)
  parts.push(`  task_name=${store.taskName}`)
  if (store.selectedEval) parts.push(`  eval=${safeStr(store.selectedEval)}`)

  const ds = store.selectedDatasets
  if (ds.forget) parts.push(`  data/datasets@data.forget=${safeStr(ds.forget)}`)
  if (ds.retain) parts.push(`  data/datasets@data.retain=${safeStr(ds.retain)}`)
  if (ds.edit) parts.push(`  data/datasets@data.edit=${safeStr(ds.edit)}`)
  if (ds.train) parts.push(`  data/datasets@data.train=${safeStr(ds.train)}`)

  for (const [k, v] of Object.entries(store.overrides)) {
    parts.push(`  ${k}=${safeStr(v)}`)
  }

  return parts.join(' \\\n')
})

function copy() {
  navigator.clipboard.writeText(commandText.value).then(() => {
    ElMessage.success(t('copilot.copySuccess'))
  })
}

async function startRun() {
  runnerStore.clearLogs()
  const body = {
    mode: store.mode,
    model: store.selectedModel,
    trainer: store.mode === 'eval' ? undefined : store.selectedTrainer,
    experiment: store.selectedExperiment || undefined,
    task_name: store.taskName,
    overrides: store.overrides,
    eval_suite: store.selectedEval || undefined,
    gpu: store.gpu,
  }
  const res = await runnerApi.start(body) as { ok?: boolean; error?: string; command?: string }
  if (res.ok) {
    runnerStore.running = true
    runnerStore.command = res.command || ''
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
