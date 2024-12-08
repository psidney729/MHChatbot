import axios from 'axios'

const API_URL = import.meta.env.VITE_BACKEND_API_URL

class ModuleService {
  async send_user_message(user_message: String) {
    const response = await axios.post(API_URL + 'modules/chatbotllm', user_message)
    return response.data
  }

  async get_chat_history() {
    const response = await axios.get(API_URL + 'modules/chathistory')
    return response.data.response
  }

  async semantic_search(query: string) {
    const response = await axios.post(API_URL + 'modules/semanticsearch', query)
    console.log(response.data.response)
    return response.data.response
  }

  async predict_response_type(query: any){
    const response = await axios.post(API_URL + 'modules/mlclassification', query)
    console.log(response.data.response)
    return response.data.response
  }
}

export default new ModuleService()
