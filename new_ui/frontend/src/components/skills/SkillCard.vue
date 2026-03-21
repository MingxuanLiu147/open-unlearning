<template>
  <div class="ks-card skill-card" :class="{ selected: isSelected }" @click="$emit('select', skill)">
    <div class="skill-header">
      <span class="skill-name">{{ skill.name }}</span>
      <span class="mode-badge" :class="skill.mode">{{ skill.mode }}</span>
    </div>
    <p class="skill-desc">{{ skill.description }}</p>
    <div class="skill-meta">
      <el-tag v-for="tag in (skill.tags || [])" :key="tag" size="small" effect="plain" round>{{ tag }}</el-tag>
      <span class="step-count">{{ (skill.steps || []).length }} {{ $t('skills.steps') }}</span>
    </div>
    <div class="skill-actions">
      <el-button size="small" type="primary" @click.stop="$emit('run', skill)">{{ $t('skills.run') }}</el-button>
      <el-button size="small" @click.stop="$emit('edit', skill)">{{ $t('skills.edit') }}</el-button>
      <el-button size="small" type="danger" text @click.stop="$emit('delete', skill)">{{ $t('skills.delete') }}</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ skill: any; isSelected: boolean }>()
defineEmits(['select', 'run', 'edit', 'delete'])
</script>

<style scoped>
.skill-card { margin-bottom: 8px; }
.skill-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.skill-name { font-weight: 600; font-size: 14px; }
.skill-desc { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }
.skill-meta { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.step-count { font-size: 11px; color: var(--text-muted); margin-left: auto; }
.skill-actions { display: flex; gap: 4px; }
</style>
