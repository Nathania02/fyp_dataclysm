import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import Welcome from '../views/Welcome.vue'
import Login from '../views/Login.vue'
import Signup from '../views/Signup.vue'
import Dashboard from '../views/Dashboard.vue'
import NewRun from '../views/NewRun.vue'
import RunDetails from '../views/RunDetails.vue'
import Notifications from '../views/Notifications.vue'
import CompareRuns from '../views/CompareRuns.vue'

const routes = [
  {
    path: '/welcome',
    name: 'Welcome',
    component: Welcome,
    meta: { requiresAuth: false, hideForAuth: false }
  },
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { requiresAuth: false, hideForAuth: true }
  },
  {
    path: '/signup',
    name: 'Signup',
    component: Signup,
    meta: { requiresAuth: false, hideForAuth: true }
  },
  {
    path: '/',
    name: 'Dashboard',
    component: Dashboard,
    meta: { requiresAuth: true }
  },
  {
    path: '/new-run',
    name: 'NewRun',
    component: NewRun,
    meta: { requiresAuth: true }
  },
  {
    path: '/runs/:id',
    name: 'RunDetails',
    component: RunDetails,
    meta: { requiresAuth: true }
  },
  {
    path: '/notifications',
    name: 'Notifications',
    component: Notifications,
    meta: { requiresAuth: true }
  },
  {
    path: '/compare-runs',
    name: 'CompareRuns',
    component: CompareRuns,
    meta: { requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  // Fetch user if authenticated but user data not loaded
  if (authStore.isAuthenticated && !authStore.user) {
    await authStore.fetchUser()
  }
  
  // If route requires auth and user is not authenticated, redirect to welcome
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    next('/welcome')
  } 
  // If user is authenticated and tries to access login/signup, redirect to dashboard
  else if (to.meta.hideForAuth && authStore.isAuthenticated) {
    next('/')
  } 
  // Allow navigation
  else {
    next()
  }
})

export default router