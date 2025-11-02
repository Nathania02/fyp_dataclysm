import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import Login from '../views/Login.vue'
import Signup from '../views/Signup.vue'
import Dashboard from '../views/Dashboard.vue'
import NewRun from '../views/NewRun.vue'
import RunDetails from '../views/RunDetails.vue'
import Notifications from '../views/Notifications.vue'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { requiresAuth: false }
  },
  {
    path: '/signup',
    name: 'Signup',
    component: Signup,
    meta: { requiresAuth: false }
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
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// router.beforeEach(async (to, from, next) => {
//   const authStore = useAuthStore()
  
//   if (to.meta.requiresAuth && !authStore.isAuthenticated) {
//     next('/login')
//   } else if (!to.meta.requiresAuth && authStore.isAuthenticated) {
//     next('/')
//   } else {
//     if (authStore.isAuthenticated && !authStore.user) {
//       await authStore.fetchUser()
//     }
//     next()
//   }
// })


router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  // --- Temporarily commented out for layout checks ---
//   if (to.meta.requiresAuth && !authStore.isAuthenticated) {
//     next('/login')
//   } else if (!to.meta.requiresAuth && authStore.isAuthenticated) {
//     next('/')
//   } else {
  // --- End of temporary comments ---

    if (authStore.isAuthenticated && !authStore.user) {
      await authStore.fetchUser()
    }
    next() // <-- This now runs every time
//   }
})

export default router