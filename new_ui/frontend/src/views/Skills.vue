<template>
  <div class="skills-view">
    <div class="skills-header">
      <h2>{{ $t('skills.title') }}</h2>
      <el-button type="primary" size="small" @click="createNew">{{ $t('skills.create') }}</el-button>
    </div>

    <el-row :gutter="16">
      <el-col :span="10">
        <SkillCard
          v-for="s in skills"
          :key="s.id"
          :skill="s"
          :is-selected="selected?.id === s.id"
          @select="selected = s"
          @edit="selected = s"
          @run="runSkill(s)"
          @delete="deleteSkill(s)"
        />
        <div v-if="skills.length === 0" class="empty">{{ $t('skills.noSkills') }}</div>
      </el-col>
      <el-col :span="14">
        <SkillEditor :skill="selected" @saved="loadSkills" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { skillsApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'
import SkillCard from '@/components/skills/SkillCard.vue'
import SkillEditor from '@/components/skills/SkillEditor.vue'

const skills = ref<any[]>([])
const selected = ref<any>(null)
const experimentStore = useExperimentStore()

async function loadSkills() {
  skills.value = await skillsApi.list()
}

function createNew() {
  selected.value = { name: '', description: '', mode: 'unlearn', tags: [], steps: [] }
}

async function runSkill(s: any) {
  if (s.steps && s.steps.length > 0) {
    const first = s.steps[0]
    experimentStore.applyConfig({
      mode: s.mode,
      model: first.model,
      trainer: first.trainer,
      datasets: first.datasets || {},
      eval: first.eval,
      params: first.params || {},
    })
    ElMessage.success('Applied first step config to Workshop')
  }
}

async function deleteSkill(s: any) {
  await ElMessageBox.confirm(`Delete "${s.name}"?`, 'Confirm')
  await skillsApi.delete(s.id)
  if (selected.value?.id === s.id) selected.value = null
  loadSkills()
}

onMounted(loadSkills)
</script>

<style scoped>
.skills-view { max-width: 1200px; margin: 0 auto; }
.skills-header {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;
}
h2 { font-size: 18px; font-weight: 600; }
.empty { color: var(--text-muted); text-align: center; padding: 40px; }
</style>
