<template>
  <div class="path-config">
    <div class="section-title">{{ $t('workshop.pathConfig') }}</div>

    <el-form label-position="top" size="small">
      <el-form-item :label="$t('workshop.modelPathHint')">
        <el-input
          v-model="store.modelPath"
          clearable
          :placeholder="$t('workshop.modelPathPh')"
        />
      </el-form-item>
      <el-form-item :label="$t('workshop.tokenizerPathHint')">
        <el-input
          v-model="store.tokenizerPath"
          clearable
          :placeholder="$t('workshop.tokenizerPathPh')"
        />
      </el-form-item>
      <el-form-item :label="$t('workshop.outputDir')">
        <el-input
          v-model="store.outputDir"
          clearable
          :placeholder="$t('workshop.outputDirPh')"
        />
      </el-form-item>

      <el-collapse>
        <el-collapse-item :title="$t('workshop.advancedModelYaml')" name="yaml">
          <el-form-item :label="$t('workshop.yamlConfigName')">
            <el-input v-model="yamlName" clearable placeholder="my_custom_model" />
          </el-form-item>
          <el-form-item :label="$t('workshop.yamlModelId')">
            <el-input v-model="yamlPath" clearable placeholder="org/model-id" />
          </el-form-item>
          <el-form-item :label="$t('workshop.yamlDtype')">
            <el-radio-group v-model="yamlDtype">
              <el-radio-button value="bfloat16">bfloat16</el-radio-button>
              <el-radio-button value="float16">float16</el-radio-button>
              <el-radio-button value="float32">float32</el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item :label="$t('workshop.yamlChatTemplate')">
            <el-switch v-model="yamlChat" />
          </el-form-item>
          <el-button type="primary" :loading="creating" @click="createYaml">
            {{ $t('workshop.generateYaml') }}
          </el-button>
          <p class="hint">{{ $t('workshop.modelCompatHint') }}</p>
        </el-collapse-item>
      </el-collapse>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useExperimentStore } from '@/stores/experiment'
import { configApi } from '@/api'

const store = useExperimentStore()
const yamlName = ref('')
const yamlPath = ref('')
const yamlDtype = ref('bfloat16')
const yamlChat = ref(true)
const creating = ref(false)

async function createYaml() {
  if (!yamlName.value.trim() || !yamlPath.value.trim()) {
    ElMessage.warning('Please fill config name and model id')
    return
  }
  creating.value = true
  try {
    const res = await configApi.createModel({
      name: yamlName.value.trim(),
      path: yamlPath.value.trim(),
      dtype: yamlDtype.value,
      apply_chat_template: yamlChat.value,
    }) as { ok?: boolean; error?: string; model?: string }
    if (res.ok && res.model) {
      ElMessage.success('Model config created')
      store.selectedModel = res.model
      const list = await configApi.getModels()
      store.setModelCatalog(list)
    } else {
      ElMessage.error(res.error || 'Failed')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || 'Failed')
  } finally {
    creating.value = false
  }
}
</script>

<style scoped>
.path-config {
  margin-bottom: 16px;
  padding: 14px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.4;
}
</style>
