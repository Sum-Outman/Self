import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout/Layout';
import PrivateRoute from './components/Auth/PrivateRoute';
import AdminRoute from './components/Auth/AdminRoute';
import { GlobalErrorBoundary } from './components/ErrorBoundary/ErrorBoundary';

// 懒加载页面组件
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const ForgotPasswordPage = lazy(() => import('./pages/ForgotPasswordPage'));
const ResetPasswordPage = lazy(() => import('./pages/ResetPasswordPage'));
const VerifyEmailPage = lazy(() => import('./pages/VerifyEmailPage'));
const TermsPage = lazy(() => import('./pages/TermsPage'));
const PrivacyPage = lazy(() => import('./pages/PrivacyPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const ChatPage = lazy(() => import('./pages/ChatPage'));
const TrainingPage = lazy(() => import('./pages/TrainingPage'));
const HardwarePage = lazy(() => import('./pages/HardwarePage'));
const RobotManagementPage = lazy(() => import('./pages/RobotManagementPage'));
const KnowledgePage = lazy(() => import('./pages/KnowledgePage'));
const MemoryPage = lazy(() => import('./pages/MemoryPage'));
const AdminPage = lazy(() => import('./pages/AdminPage'));
const ApiKeysPage = lazy(() => import('./pages/ApiKeysPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const ProfessionalCapabilitiesPage = lazy(() => import('./pages/ProfessionalCapabilitiesPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const MultimodalConceptPage = lazy(() => import('./pages/MultimodalConceptPage'));
const AutonomousModePage = lazy(() => import('./pages/AutonomousModePage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

// 加载回退组件
const LoadingFallback = () => (
  <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
    <div className="text-center">
      <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-600 dark:border-gray-400"></div>
      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">加载中...</p>
    </div>
  </div>
);

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ThemeProvider>
        <AuthProvider>
          <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200">
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#1f2937', // gray-800
                  color: '#f9fafb', // gray-50
                },
                success: {
                  duration: 3000,
                  iconTheme: {
                    primary: '#475569', // gray-600
                    secondary: '#f9fafb', // gray-50
                  },
                },
                error: {
                  duration: 4000,
                  iconTheme: {
                    primary: '#1e293b', // gray-800
                    secondary: '#f9fafb', // gray-50
                  },
                },
              }}
            />
            

            <Suspense fallback={<LoadingFallback />}>
              <GlobalErrorBoundary>
                <Routes>
                  {/* 公开路由 */}
                  <Route path="/login" element={<LoginPage />} />
                  <Route path="/register" element={<RegisterPage />} />
                  <Route path="/forgot-password" element={<ForgotPasswordPage />} />
                  <Route path="/reset-password" element={<ResetPasswordPage />} />
                  <Route path="/verify-email" element={<VerifyEmailPage />} />
                  <Route path="/terms" element={<TermsPage />} />
                  <Route path="/privacy" element={<PrivacyPage />} />
                  
                  {/* 需要认证的路由 */}
                  <Route path="/" element={
                    <PrivateRoute>
                      <Layout />
                    </PrivateRoute>
                  }>
                    <Route index element={<Navigate to="/dashboard" replace />} />
                    <Route path="dashboard" element={<DashboardPage />} />
                    <Route path="chat" element={<ChatPage />} />
                    <Route path="training" element={
                      <AdminRoute>
                        <TrainingPage />
                      </AdminRoute>
                    } />
                    <Route path="hardware" element={<HardwarePage />} />
                    <Route path="robot-management" element={<RobotManagementPage />} />
                    <Route path="knowledge" element={
                      <AdminRoute>
                        <KnowledgePage />
                      </AdminRoute>
                    } />
                    <Route path="memory" element={
                      <AdminRoute>
                        <MemoryPage />
                      </AdminRoute>
                    } />
                    <Route path="api-keys" element={<ApiKeysPage />} />
                    <Route path="settings" element={<SettingsPage />} />
                    <Route path="profile" element={<ProfilePage />} />
                    <Route path="professional-capabilities" element={<ProfessionalCapabilitiesPage />} />
                    <Route path="multimodal-concept" element={<MultimodalConceptPage />} />
                    <Route path="autonomous-mode" element={<AutonomousModePage />} />
                    
                    {/* 管理员路由 */}
                    <Route path="admin" element={
                      <AdminRoute>
                        <AdminPage />
                      </AdminRoute>
                    } />
                  </Route>
                  
                  {/* 404页面 */}
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </GlobalErrorBoundary>
            </Suspense>
          </div>
        </AuthProvider>
      </ThemeProvider>
    </Router>
  );
}

export default App;