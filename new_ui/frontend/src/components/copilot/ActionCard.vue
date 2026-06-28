<template>
  <div class="action-card" :class="[action.status, action.type]">
    <div class="action-indicator"></div>
    <div class="action-body">
      <div class="action-header">
        <span class="action-type-badge">
          {{ action.type === 'apply_config' ? $t('copilot.actionApplyConfig') : $t('copilot.actionNavigate') }}
        </span>
        <span v-if="action.status !== 'pending'" class="action-status-tag" :class="action.status">
          {{ action.status === 'applied' ? $t('copilot.applied') : $t('copilot.rejected') }}
        </span>
      </div>
      <div class="action-preview">{{ action.preview }}</div>
      <div class="action-payload-section">
        <button class="toggle-detail" @click="showPayload = !showPayload">
          <svg
            viewBox="0 0 24 24" width="12" height="12" fill="none"
            stroke="currentColor" stroke-width="2"
            :class="{ rotated: showPayload }"
          >
            <path d="M9 18l6-6-6-6"/>
          </svg>
          {{ showPayload ? '收起详情' : '查看详情' }}
        </button>
        <div class="action-payload" v-if="showPayload">
          <pre><code>{{ formattedPayload }}</code></pre>
        </div>
      </div>
      <div class="action-footer" v-if="action.status === 'pending'">
        <el-button type="success" size="small" @click="$emit('apply', action)" class="apply-btn">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M20 6L9 17l-5-5"/>
          </svg>
          {{ $t('copilot.apply') }}
        </el-button>
        <el-button size="small" @click="$emit('reject', action.id)" class="reject-btn">
          {{ $t('copilot.reject') }}
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { AgentAction } from '@/stores/agent'

const props = defineProps<{ action: AgentAction }>()
defineEmits<{
  (e: 'apply', action: AgentAction): void
  (e: 'reject', id: string): void
}>()

const showPayload = ref(false)

const formattedPayload = computed(() => {
  try {
    return JSON.stringify(props.action.payload, null, 2)
  } catch {
    return String(props.action.payload)
  }
})
</script>

<style scoped>
.action-card {
  display: flex;
  margin: 10px 0;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--border-color);
  background: var(--bg-card);
  transition: all var(--transition-fast);
}
.action-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--accent-primary);
}

.action-indicator {
  width: 4px;
  flex-shrink: 0;
  transition: background var(--transition-fast);
}
.action-card.pending.apply_config .action-indicator {
  background: linear-gradient(180deg, #3b82f6, #6366f1);
}
.action-card.pending.navigate_view .action-indicator {
  background: linear-gradient(180deg, #8b5cf6, #a855f7);
}
.action-card.applied .action-indicator {
  background: linear-gradient(180deg, #10b981, #34d399);
}
.action-card.rejected .action-indicator {
  background: var(--text-muted);
}

.action-card.applied {
  border-color: var(--accent-success);
  background: var(--accent-success-soft);
}
.action-card.rejected {
  opacity: 0.55;
}

.action-body {
  flex: 1;
  padding: 12px 14px;
  min-width: 0;
}

.action-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.action-type-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 999px;
  background: var(--accent-primary-soft);
  color: var(--accent-primary);
  letter-spacing: 0.3px;
}
.action-card.applied .action-type-badge {
  background: var(--accent-success-soft);
  color: var(--accent-success);
}
.action-status-tag {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
}
.action-status-tag.applied {
  background: var(--accent-success-soft);
  color: var(--accent-success);
}
.action-status-tag.rejected {
  background: var(--bg-card-hover);
  color: var(--text-muted);
}

.action-preview {
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-primary);
  font-weight: 500;
}

.action-payload-section {
  margin-top: 8px;
}
.toggle-detail {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
  background: none;
  border: none;
  cursor: pointer;
  padding: 2px 0;
  transition: color var(--transition-fast);
}
.toggle-detail:hover {
  color: var(--accent-primary);
}
.toggle-detail svg {
  transition: transform var(--transition-fast);
}
.toggle-detail svg.rotated {
  transform: rotate(90deg);
}

.action-payload {
  margin-top: 6px;
  border-radius: 6px;
  overflow: hidden;
}
.action-payload pre {
  margin: 0;
  background: var(--bg-code);
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 11px;
  line-height: 1.5;
}
.action-payload code {
  font-family: var(--font-mono);
  color: var(--text-secondary);
  white-space: pre-wrap;
  word-break: break-all;
}

.action-footer {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
.apply-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-weight: 600;
}
.reject-btn {
  color: var(--text-muted);
}
</style>
