import { createBrowserRouter } from 'react-router-dom'
import ProtectedRoute from './routes/protectedroute'
import ErrorPage from './error-page'
import Home, { loader as homeLoader } from './routes/home'
import Login from './routes/login'
import { Profile } from './routes/profile'
import Register from './routes/register'
import Root from './routes/root'
import SSOLogin, { loader as ssoLoader } from './routes/sso.login'
import Users, { loader as usersLoader } from './routes/users'
import LLMchatbot from './routes/llm'
import MLclassification from './routes/ml'
import SemanticSearch from './routes/ss'

export const routes = [
  {
    path: '/',
    element: <Root />,
    errorElement: <ErrorPage />,
    children: [
      {
        element: <ProtectedRoute />,
        children: [
          {
            path: 'profile',
            element: <Profile />,
          },
          {
            path: 'users',
            element: <Users />,
            loader: usersLoader,
          },
          {
            path: 'llm',
            element: <LLMchatbot />,
          },
          {
            path: 'ml',
            element: <MLclassification />,
          },
          {
            path: 'ss',
            element: <SemanticSearch />,
          }
        ],
      },
      { index: true, element: <Home />, loader: homeLoader },
      {
        path: 'sso-login-callback',
        element: <SSOLogin />,
        loader: ssoLoader,
      },
      {
        path: 'login',
        element: <Login />,
      },
      {
        path: 'register',
        element: <Register />,
      },
    ],
  },
]

export const router = createBrowserRouter(routes)
