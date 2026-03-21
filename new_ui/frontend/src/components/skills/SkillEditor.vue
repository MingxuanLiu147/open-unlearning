<template>
  <div class="editor" v-if="skill">
    <h3>{{ skill.name || 'New Skill' }}</h3>
    <el-form label-position="top" size="small">
      <el-form-item label="Name">
        <el-input v-model="draft.name" />
      </el-form-item>
      <el-form-item label="Description">
        <el-input v-model="draft.description" type="textarea" :rows="2" />
      </el-form-item>
      <el-form-item label="Mode">
        <el-select v-model="draft.mode">
          <el-option value="unlearn" label="Unlearn" />
          <el-option value="inject" label="Inject" />
          <el-option value="edit" label="Edit" />
        </el-select>
      </el-form-item>
      <el-form-item label="Tags (comma separated)">
        <el-input v-model="tagsStr" />
      </el-form-item>

      <div class="steps-section">
        <h4>{{ $t('skills.steps') }} ({{ (draft.steps || []).length }})</h4>
        <div v-for="(step, i) in draft.steps" :key="i" class="step-item ks-card">
          <el-row :gutter="8">
            <el-col :span="8">
              <el-form-item label="Model"><el-input v-model="step.model" size="small" /></el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="Trainer"><el-input v-model="step.trainer" size="small" /></el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="Eval"><el-input v-model="step.eval" size="small" /></el-form-item>
            </el-col>
          </el-row>
          <el-button text size="small" type="danger" @click="draft.steps.splice(i, 1)">Remove step</el-button>
        </div>
        <el-button size="small" @click="addStep">+ Add Step</el-button>
      </div>

      <div class="editor-actions">
        <el-button type="primary" @click="save">Save</el-button>
      </div>
    </el-form>
  </div>
  <div v-else class="empty-editor">
    <p>{{ $t('skills.noSkills') }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { skillsApi } from '@/api'

const props = defineProps<{ skill: any }>()
const emit = defineEmits(['saved'])

const draft = ref<any>({})
const tagsStr = computed({
  get: () => (draft.value.tags || []).join(', '),
  set: (v: string) => { draft.value.tags = v.split(',').map((s: string) => s.trim()).filter(Boolean) },
})

watch(() => props.skill, (s) => {
  draft.value = JSON.parse(JSON.stringify(s || { name: '', description: '', mode: 'unlearn', tags: [], steps: [] }))
}, { immediate: true })

function addStep() {
  if (!draft.value.steps) draft.value.steps = []
  draft.value.steps.push({ model: '', trainer: '', datasets: {}, eval: '', params: {} })
}

async function save() {
  const id = draft.value.id
  if (id) {
    await skillsApi.update(id, draft.value)
  } else {
    await skillsApi.create(draft.value)
  }
  ElMessage.success('Saved')
  emit('saved')
}
</script>

<style scoped>
.editor { padding: 4px; }
h3 { font-size: 16px; margin-bottom: 12px; }
.steps-section { margin-top: 12px; }
.steps-section h4 { font-size: 13px; margin-bottom: 8px; }
.step-item { margin-bottom: 8px; }
.editor-actions { margin-top: 16px; }
.empty-editor { padding: 40px; text-align: center; color: var(--text-muted); }
</style>
