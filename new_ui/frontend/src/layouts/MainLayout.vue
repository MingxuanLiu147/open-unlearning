<template>
  <div class="layout">
    <!-- Topbar -->
    <header class="topbar">
      <div class="topbar-left">
        <div class="logo">
          <span class="logo-icon">🔬</span>
          <span class="logo-text">{{ $t('app.title') }}</span>
        </div>
        <nav class="nav-tabs">
          <router-link
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            class="nav-item"
            :class="{ active: $route.path === item.path }"
          >
            <el-icon :size="16"><component :is="item.icon" /></el-icon>
            <span>{{ $t(item.label) }}</span>
          </router-link>
        </nav>
      </div>
      <div class="topbar-right">
        <LangToggle />
        <ThemeToggle />
        <el-tooltip :content="$t('copilot.title')">
          <el-button circle size="small" @click="appStore.toggleCopilot"
            :type="appStore.copilotOpen ? 'primary' : 'default'">
            <el-icon :size="16"><ChatDotRound /></el-icon>
          </el-button>
        </el-tooltip>
      </div>
    </header>

    <!-- Body -->
    <div class="main-body">
      <div class="content" :class="{ 'copilot-open': appStore.copilotOpen }">
        <router-view />
      </div>
      <transition name="slide-copilot">
        <aside v-if="appStore.copilotOpen" class="copilot-sidebar">
          <CopilotPanel />
        </aside>
      </transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from '@/stores/app'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import LangToggle from '@/components/common/LangToggle.vue'
import CopilotPanel from '@/components/copilot/CopilotPanel.vue'
import { SetUp, Monitor, DataAnalysis, MagicStick, ChatDotRound } from '@element-plus/icons-vue'

const appStore = useAppStore()

const navItems = [
  { path: '/workshop', label: 'nav.workshop', icon: SetUp },
  { path: '/monitor', label: 'nav.monitor', icon: Monitor },
  { path: '/results', label: 'nav.results', icon: DataAnalysis },
  { path: '/skills', label: 'nav.skills', icon: MagicStick },
]
</script>

<style scoped>
.layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  height: var(--topbar-height);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: var(--shadow-sm);
}

.topbar-left {
  display: flex;
  align-items: center;
  gap: 32px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 16px;
  color: var(--text-primary);
  white-space: nowrap;
}
.logo-icon { font-size: 20px; }

.nav-tabs {
  display: flex;
  gap: 4px;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  transition: all var(--transition-fast);
  text-decoration: none;
}
.nav-item:hover {
  background: var(--bg-card-hover);
  color: var(--text-primary);
}
.nav-item.active {
  background: var(--accent-primary-soft);
  color: var(--accent-primary);
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.main-body {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  transition: margin-right var(--transition-normal);
}

.copilot-sidebar {
  width: var(--copilot-width);
  min-width: var(--copilot-width);
  background: var(--bg-copilot);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.slide-copilot-enter-active,
.slide-copilot-leave-active {
  transition: all var(--transition-normal);
}
.slide-copilot-enter-from,
.slide-copilot-leave-to {
  width: 0;
  min-width: 0;
  opacity: 0;
}
</style>
