import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/auth'

const ProtectedRoute = () => {
  const { user } = useAuth()

  if (!user) {
    return <Navigate to='/' replace />
  }

  return <Outlet />
}

export default ProtectedRoute
