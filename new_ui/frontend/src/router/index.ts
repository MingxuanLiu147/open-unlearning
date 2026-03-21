import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '@/layouts/MainLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: MainLayout,
      redirect: '/workshop',
      children: [
        { path: 'workshop', name: 'workshop', component: () => import('@/views/Workshop.vue') },
        { path: 'monitor', name: 'monitor', component: () => import('@/views/Monitor.vue') },
        { path: 'results', name: 'results', component: () => import('@/views/Results.vue') },
        { path: 'skills', name: 'skills', component: () => import('@/views/Skills.vue') },
      ],
    },
  ],
})

export default router
