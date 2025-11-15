<template>
  <div class="auth-container" :class="{ 'sign-up-mode': isSignupMode }">
    <div class="forms-container">
      <div class="signin-signup" >
        <!-- Sign In Form -->
        <form @submit.prevent="handleLogin" class="sign-in-form">
          <h2 class="form-title">Sign In</h2>
          <p class="form-subtitle">Welcome back to the Model Training Platform</p>
          
          <div v-if="loginError" class="error-message">
            <i class="fas fa-exclamation-circle" style="padding-left: 30%;"></i>
            {{ loginError }}
          </div>
          
          <div class="input-field">
            <i class="fas fa-envelope" style="padding-left: 30%;"></i>
            <input
              v-model="loginForm.email"
              type="email"
              placeholder="Email"
              required
            />
          </div>
          
          <div class="input-field">
            <i class="fas fa-lock" style="padding-left: 30%;"></i>
            <input
              v-model="loginForm.password"
              type="password"
              placeholder="Password"
              required
            />
          </div>
          
          <button type="submit" class="btn-submit">Login</button>
        </form>

        <!-- Sign Up Form -->
        <form @submit.prevent="handleSignup" class="sign-up-form">
          <h2 class="form-title">Sign Up</h2>
          <p class="form-subtitle">Create your account to get started</p>
          
          <div v-if="signupError" class="error-message">
            <i class="fas fa-exclamation-circle" style="padding-left: 30%;"></i>
            {{ signupError }}
          </div>
          
          <div class="input-field">
            <i class="fas fa-envelope" style="padding-left: 30%;"></i>
            <input
              v-model="signupForm.email"
              type="email"
              placeholder="Email"
              required
            />
          </div>
          
          <div class="input-field">
            <i class="fas fa-lock" style="padding-left: 30%;"></i>
            <input
              v-model="signupForm.password"
              type="password"
              placeholder="Password"
              required
              minlength="8"
            />
          </div>

          <div class="input-field">
            <i class="fas fa-lock" style="padding-left: 30%;"></i>
            <input
              v-model="signupForm.confirmPassword"
              type="password"
              placeholder="Confirm Password"
              required
              minlength="8"
            />
          </div>
          
          <div class="input-field select-field">
            <i class="fas fa-user-tag" style="padding-left: 30%;"></i>
            <select v-model="signupForm.role" required>
              <option value="" disabled>Select Role</option>
              <option value="data_scientist">Data Scientist</option>
              <option value="clinician">Clinician</option>
            </select>
          </div>
          
          <button type="submit" class="btn-submit">Sign Up</button>
        </form>
      </div>
    </div>

    <div class="panels-container">
      <!-- Left Panel -->
      <div class="panel left-panel">
        <div class="panel-content" style="padding-bottom:30%">
          <h3>New here?</h3>
          <p>Join SingHealth's collaborative model training platform for sepsis phenotyping research and work with leading healthcare professionals today!</p>
          <button @click="toggleMode" class="btn-transparent" type="button">Sign Up</button>
        </div>
        <!-- <img src="../assets/login-hero.jpg" class="panel-image" alt="Medical Illustration" /> -->
      </div>

      <!-- Right Panel -->
      <div class="panel right-panel">
        <div class="panel-content" style="padding-bottom:30%">
          <h3>One of us?</h3>
          <p>Sign in to continue your research and collaborate with your team on advanced phenotyping models</p>
          <button @click="toggleMode" class="btn-transparent" type="button">Sign In</button>
        </div>
        <!-- <img src="../assets/signup-hero.jpg" class="panel-image" alt="Medical Illustration" /> -->
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const isSignupMode = ref(false)

const loginForm = ref({
  email: '',
  password: '',
  email_icon: '',
  pwd_icon: ''
})

const signupForm = ref({
  email: '',
  password: '',
  confirmPassword: '',
  role: '',
  email_icon: '',
  pwd_icon: '',
  cpwd_icon: '',
  role_icon: ''

})

const loginError = ref('')
const signupError = ref('')

// Set initial mode based on route
onMounted(() => {
  if (route.path === '/signup') {
    isSignupMode.value = true
  }
})

const toggleMode = () => {
  isSignupMode.value = !isSignupMode.value
  loginError.value = ''
  signupError.value = ''
  
  // Update URL without full navigation
  if (isSignupMode.value) {
    router.replace('/signup')
  } else {
    router.replace('/login')
  }
}

const handleLogin = async () => {
  try {
    loginError.value = ''
    await authStore.login(loginForm.value.email, loginForm.value.password)
    router.push('/')
  } catch (err) {
    loginError.value = err.response?.data?.detail || 'Login failed. Please check your credentials.'
  }
}

const handleSignup = async () => {
  signupError.value = ''

  // Password validation
  if (signupForm.value.password !== signupForm.value.confirmPassword) {
    signupError.value = 'Passwords do not match.'
    return
  }

  if (signupForm.value.password.length < 8) {
    signupError.value = 'Password must be at least 8 characters long.'
    return
  }
  
  const specialCharRegex = /[!@#$%^&*(),.?":{}|<>]/
  if (!specialCharRegex.test(signupForm.value.password)) {
    signupError.value = 'Password must contain at least one special character.'
    return
  }

  try {
    await authStore.signup(
      signupForm.value.email,
      signupForm.value.password,
      signupForm.value.role
    )
    router.push('/')
  } catch (err) {
    signupError.value = err.response?.data?.detail || 'Signup failed. Please try again.'
  }
}
</script>

<style scoped>
* {
  box-sizing: border-box;
}

.auth-container {
  position: relative;
  width: 100%;
  min-height: 100vh;
  background: #f8f9fa;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.forms-container {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
}

.signin-signup {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  left: 75%;
  width: 50%;
  max-width: 520px;
  transition: 1s 0.7s ease-in-out;
  display: grid;
  grid-template-columns: 1fr;
  z-index: 5;
}

form {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  padding: 2rem 3rem;
  transition: all 0.2s 0.7s;
  overflow: hidden;
  grid-column: 1 / 2;
  grid-row: 1 / 2;
}

form.sign-up-form {
  opacity: 0;
  z-index: 1;
}

form.sign-in-form {
  z-index: 2;
}

.form-title {
  font-size: 2.2rem;
  color: #2c3e50;
  margin-bottom: 0.5rem;
  font-weight: 700;
  letter-spacing: -0.5px;
}

.form-subtitle {
  font-size: 0.95rem;
  color: #7f8c8d;
  margin-bottom: 2rem;
  text-align: center;
  line-height: 1.5;
  max-width: 350px;
}

.error-message {
  width: 100%;
  max-width: 420px;
  padding: 0.875rem 1.25rem;
  background: #fff5f5;
  border: 1px solid #feb2b2;
  border-radius: 12px;
  color: #c53030;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  font-weight: 500;
  animation: slideDown 0.3s ease-out;
  box-shadow: 0 2px 8px rgba(197, 48, 48, 0.1);
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.input-field {
  width: 100%;
  max-width: 420px;
  background: white;
  margin: 0.625rem 0;
  height: 56px;
  border-radius: 12px;
  display: grid;
  grid-template-columns: 50px 1fr;
  padding: 0;
  border: 2px solid #e2e8f0;
  transition: all 0.3s ease;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.input-field:focus-within {
  border-color: #2B515D;
  box-shadow: 0 0 0 3px rgba(43, 81, 93, 0.1);
  background: #ffffff;
}

.input-field i {
  text-align: center;
  line-height: 56px;
  color: #94a3b8;
  font-size: 1.1rem;
  transition: 0.3s;
}

.input-field:focus-within i {
  color: #2B515D;
}

.input-field input,
.input-field select {
  background: none;
  outline: none;
  border: none;
  line-height: 56px;
  font-weight: 500;
  font-size: 0.95rem;
  color: #2c3e50;
  width: 100%;
  padding-right: 1rem;
}

.input-field.select-field select {
  cursor: pointer;
  padding-right: 2.5rem;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394a3b8' d='M10.293 3.293L6 7.586 1.707 3.293A1 1 0 00.293 4.707l5 5a1 1 0 001.414 0l5-5a1 1 0 10-1.414-1.414z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 1rem center;
}

.input-field input::placeholder {
  color: #94a3b8;
  font-weight: 400;
}

.input-field select option {
  background: white;
  color: #2c3e50;
  padding: 0.5rem;
}

.btn-submit {
  width: 100%;
  max-width: 420px;
  background: linear-gradient(135deg, #2B515D 0%, #3d6b7a 100%);
  border: none;
  outline: none;
  height: 52px;
  border-radius: 12px;
  color: white;
  text-transform: uppercase;
  font-weight: 600;
  font-size: 0.95rem;
  letter-spacing: 0.5px;
  margin-top: 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 14px rgba(43, 81, 93, 0.25);
}

.btn-submit:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(43, 81, 93, 0.35);
  background: linear-gradient(135deg, #234350 0%, #2f5a68 100%);
}

.btn-submit:active {
  transform: translateY(0);
  box-shadow: 0 2px 8px rgba(43, 81, 93, 0.25);
}

.panels-container {
  position: absolute;
  height: 100%;
  width: 100%;
  top: 0;
  left: 0;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
}

.auth-container:before {
  content: "";
  position: absolute;
  height: 2000px;
  width: 2000px;
  top: -10%;
  right: 48%;
  transform: translateY(-50%);
  background: linear-gradient(135deg, #2B515D 0%, #1a3940 100%);
  transition: 1.8s ease-in-out;
  border-radius: 50%;
  z-index: 6;
  box-shadow: 0 8px 32px rgba(43, 81, 93, 0.3);
}

.panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  z-index: 7;
  position: relative;
  padding: 3rem 10%;
}

.left-panel {
  pointer-events: all;
}

.right-panel {
  pointer-events: none;
}

.panel-content {
  color: white;
  transition: transform 0.9s ease-in-out;
  transition-delay: 0.6s;
  z-index: 2;
  max-width: 450px;
}

.panel-content h3 {
  font-weight: 700;
  font-size: 2.25rem;
  margin-bottom: 1rem;
  line-height: 1.2;
  letter-spacing: -0.5px;
}

.panel-content p {
  font-size: 1.05rem;
  padding: 0;
  line-height: 1.7;
  opacity: 0.95;
  margin-bottom: 2rem;
  font-weight: 400;
}

.btn-transparent {
  background: transparent;
  border: 2px solid white;
  width: 160px;
  height: 52px;
  color: white;
  border-radius: 12px;
  font-weight: 600;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-transparent:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.btn-transparent:active {
  transform: translateY(0);
}

.panel-image {
  position: absolute;
  width: 100%;
  max-width: 480px;
  opacity: 0.5;
  pointer-events: none;
  transition: transform 1.1s ease-in-out;
  transition-delay: 0.4s;
  object-fit: contain;
  filter: grayscale(100%) brightness(1.5);
  
}

.right-panel .panel-image,
.right-panel .panel-content {
  transform: translateX(800px);
}

/* ANIMATION */
.auth-container.sign-up-mode:before {
  transform: translate(100%, -50%);
  right: 52%;
}

.auth-container.sign-up-mode .left-panel .panel-image,
.auth-container.sign-up-mode .left-panel .panel-content {
  transform: translateX(-800px);
}

.auth-container.sign-up-mode .signin-signup {
  left: 25%;
}

.auth-container.sign-up-mode form.sign-up-form {
  opacity: 1;
  z-index: 2;
}

.auth-container.sign-up-mode form.sign-in-form {
  opacity: 0;
  z-index: 1;
}

.auth-container.sign-up-mode .right-panel .panel-image,
.auth-container.sign-up-mode .right-panel .panel-content {
  transform: translateX(0%);
}

.auth-container.sign-up-mode .left-panel {
  pointer-events: none;
}

.auth-container.sign-up-mode .right-panel {
  pointer-events: all;
}

/* Responsive Design */
@media (max-width: 1024px) {
  .signin-signup {
    max-width: 460px;
  }
  
  form {
    padding: 2rem 2.5rem;
  }
}

@media (max-width: 870px) {
  .auth-container {
    min-height: 800px;
    height: 100vh;
  }

  .signin-signup {
    width: 90%;
    max-width: 440px;
    top: 95%;
    transform: translate(-50%, -100%);
    transition: 1s 0.8s ease-in-out;
  }

  .signin-signup,
  .auth-container.sign-up-mode .signin-signup {
    left: 50%;
  }

  form {
    padding: 1.5rem 2rem;
  }

  .panels-container {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr 2fr 1fr;
  }

  .panel {
    flex-direction: row;
    justify-content: space-around;
    align-items: center;
    padding: 2rem 5%;
    grid-column: 1 / 2;
    overflow: hidden;
  }

  .right-panel {
    grid-row: 3 / 4;
  }

  .left-panel {
    grid-row: 1 / 2;
  }

  .panel-image {
    width: 180px;
    max-width: 180px;
    position: relative;
    opacity: 0.2;
  }

  .panel-content {
    max-width: none;
    text-align: left;
    padding-right: 2rem;
  }

  .panel-content h3 {
    font-size: 1.75rem;
  }

  .panel-content p {
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }

  .btn-transparent {
    width: 140px;
    height: 46px;
    font-size: 0.875rem;
  }

  .auth-container:before {
    width: 1500px;
    height: 1500px;
    transform: translateX(-50%);
    left: 30%;
    bottom: 68%;
    right: initial;
    top: initial;
    transition: 2s ease-in-out;
  }

  .auth-container.sign-up-mode:before {
    transform: translate(-50%, 100%);
    bottom: 32%;
    right: initial;
  }

  .auth-container.sign-up-mode .left-panel .panel-image,
  .auth-container.sign-up-mode .left-panel .panel-content {
    transform: translateY(-300px);
  }

  .auth-container.sign-up-mode .right-panel .panel-image,
  .auth-container.sign-up-mode .right-panel .panel-content {
    transform: translateY(0px);
  }

  .right-panel .panel-image,
  .right-panel .panel-content {
    transform: translateY(300px);
  }

  .auth-container.sign-up-mode .signin-signup {
    top: 5%;
    transform: translate(-50%, 0);
  }
}

@media (max-width: 570px) {
  form {
    padding: 1.5rem 1.5rem;
  }

  .form-title {
    font-size: 1.875rem;
  }

  .form-subtitle {
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
  }

  .input-field {
    height: 52px;
    grid-template-columns: 45px 1fr;
  }

  .input-field i {
    line-height: 52px;
  }

  .input-field input,
  .input-field select {
    line-height: 52px;
    font-size: 0.9rem;
  }

  .btn-submit {
    height: 48px;
    font-size: 0.875rem;
  }

  .panel-image {
    display: none;
  }

  .panel-content {
    padding: 0 1rem;
    text-align: center;
  }

  .panel-content h3 {
    font-size: 1.5rem;
  }

  .panel-content p {
    font-size: 0.875rem;
  }

  .auth-container {
    padding: 1rem;
  }

  .auth-container:before {
    bottom: 72%;
    left: 50%;
  }

  .auth-container.sign-up-mode:before {
    bottom: 28%;
    left: 50%;
  }
}
</style>