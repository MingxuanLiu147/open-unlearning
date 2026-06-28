<template>
  <div class="data-upload">
    <div class="section-title">{{ $t('workshop.customData') }}</div>
    <p class="schema-hint">{{ schemaHint }}</p>

    <!-- Unlearn 模式：Forget + Retain 两个上传 -->
    <div v-show="isUnlearn" class="upload-group">
      <div class="up-row">
        <span class="lbl">Forget</span>
        <el-upload
          :auto-upload="false"
          :limit="1"
          accept=".json,.jsonl,.txt"
          :on-change="(f: any) => onFile('unlearn', 'forget', f.raw)"
        >
          <el-button type="primary" plain size="small">{{ $t('workshop.chooseFile') }}</el-button>
        </el-upload>
        <span v-if="status.forget" class="stat">{{ status.forget }}</span>
      </div>
      <div class="up-row">
        <span class="lbl">Retain</span>
        <el-upload
          :auto-upload="false"
          :limit="1"
          accept=".json,.jsonl,.txt"
          :on-change="(f: any) => onFile('unlearn', 'retain', f.raw)"
        >
          <el-button type="primary" plain size="small">{{ $t('workshop.chooseFile') }}</el-button>
        </el-upload>
        <span v-if="status.retain" class="stat">{{ status.retain }}</span>
      </div>
    </div>

    <!-- Inject 模式：训练集上传 -->
    <div v-show="isInject" class="upload-group">
      <div class="up-row">
        <span class="lbl">{{ $t('workshop.trainFile') }}</span>
        <el-upload
          :auto-upload="false"
          :limit="1"
          accept=".json,.jsonl,.txt"
          :on-change="(f: any) => onFile('inject', 'train', f.raw)"
        >
          <el-button type="primary" plain size="small">{{ $t('workshop.chooseFile') }}</el-button>
        </el-upload>
        <span v-if="status.train" class="stat">{{ status.train }}</span>
      </div>
    </div>

    <!-- Edit 模式：编辑集上传 -->
    <div v-show="isEdit" class="upload-group">
      <div class="up-row">
        <span class="lbl">{{ $t('workshop.editFile') }}</span>
        <el-upload
          :auto-upload="false"
          :limit="1"
          accept=".json,.jsonl,.txt"
          :on-change="(f: any) => onFile('edit', 'edit', f.raw)"
        >
          <el-button type="primary" plain size="small">{{ $t('workshop.chooseFile') }}</el-button>
        </el-upload>
        <span v-if="status.edit" class="stat">{{ status.edit }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import { useExperimentStore } from '@/stores/experiment'
import { dataApi } from '@/api'

const props = defineProps<{ mode: string }>()

const { t } = useI18n()
const store = useExperimentStore()

const status = reactive<Record<string, string>>({})

const isUnlearn = computed(() => props.mode === 'unlearn')
const isInject = computed(() => props.mode === 'inject')
const isEdit = computed(() => props.mode === 'edit')

const schemaHint = computed(() => {
  const m = props.mode
  let base = ''
  if (m === 'unlearn') base = t('workshop.schemaUnlearn')
  else if (m === 'inject') base = t('workshop.schemaInject')
  else if (m === 'edit') base = t('workshop.schemaEdit')
  if (!base) return ''
  return base + '\n' + t('workshop.schemaFreetext')
})

// mode 切换时清空状态
watch(() => props.mode, () => {
  Object.keys(status).forEach(k => delete status[k])
})

async function onFile(mode: string, purpose: string, file: File) {
  status[purpose] = '⏳ uploading…'
  try {
    const res: {
      ok: boolean
      dataset_stem?: string
      total?: number
      passed?: number
      error?: string
      errors?: string[]
    } = await dataApi.upload(file, mode, purpose)
    if (!res.ok) {
      const msg = (res.errors && res.errors.slice(0, 3).join('; ')) || res.error || 'Validation failed'
      ElMessage.error(msg)
      status[purpose] = `✗ ${msg}`
      return
    }
    status[purpose] = `✓ ${res.passed}/${res.total} records`
    if (res.dataset_stem) {
      store.selectedDatasets[purpose] = res.dataset_stem
      store.bumpDatasetList()
    }
    ElMessage.success(t('workshop.uploadSuccess'))
  } catch (e: any) {
    ElMessage.error(e?.message || 'Upload failed')
  }
}
</script>

<style scoped>
.data-upload {
  margin-top: 16px;
  padding: 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
}
.section-title {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 6px;
}
.schema-hint {
  font-size: 12px;
  color: var(--text-muted);
  margin: 0 0 12px;
  white-space: pre-wrap;
}
.upload-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.up-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.lbl {
  min-width: 80px;
  font-size: 12px;
  color: var(--text-secondary);
}
.stat {
  font-size: 12px;
  color: var(--text-muted);
}
</style>
