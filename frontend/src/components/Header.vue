<template>
  <nav class="main-nav" :class="{ 'scrolled': isScrolled }">
    <div class="nav-container">
      <div class="nav-logo" @click="goHome">
        <span class="logo-icon">
            <div class="logo-circle"></div>
            🧠
        </span>
        Phenosense
      </div>
      <div class="nav-links">
        <router-link to="/about" class="nav-link">About Us</router-link>
        <a @click="goToFeatures" class="nav-link">Features</a>
        <button @click="$router.push('/login')" class="btn btn-secondary nav-btn">Login</button>
        <button @click="$router.push('/signup')" class="btn btn-primary nav-btn">Sign Up</button>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const isScrolled = ref(false)

const handleScroll = () => {
  isScrolled.value = window.scrollY > 50
}

const goHome = () => {
  router.push('/welcome')
}

const goToFeatures = () => {
  // If already on welcome page, scroll to features
  if (router.currentRoute.value.path === '/welcome') {
    const element = document.getElementById('features')
    if (element) {
      const navHeight = 70
      const elementPosition = element.getBoundingClientRect().top + window.pageYOffset
      const offsetPosition = elementPosition - navHeight
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      })
    }
  } else {
    // Otherwise, navigate to welcome page with hash
    router.push('/welcome#features')
  }
}

onMounted(() => {
  window.addEventListener('scroll', handleScroll)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
})
</script>

<style scoped>
/* Sticky Navigation */
.main-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: rgba(43, 81, 93, 0.95);
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.main-nav.scrolled {
  background: rgba(43, 81, 93, 1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-logo {
  font-size: 1.5rem;
  font-weight: 700;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: opacity 0.2s;
}

.nav-logo:hover {
  opacity: 0.8;
}

.logo-icon {
  font-size: 1.8rem;
}

.nav-links {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.nav-link {
  color: rgba(255, 255, 255, 0.9);
  text-decoration: none;
  cursor: pointer;
  transition: color 0.2s;
  font-weight: 500;
}

.nav-link:hover {
  color: white;
}

.nav-btn {
  margin-left: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
  .nav-links {
    gap: 1rem;
  }
  
  .nav-link {
    display: none;
  }
}
</style>