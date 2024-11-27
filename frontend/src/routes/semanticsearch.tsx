import { useState } from 'react'
import {
  Box,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Container,
} from '@mui/material'
import { useSnackBar } from '../contexts/snackbar'
import moduleService from "../services/modules.service"

interface SearchResult {
  0: number
  1: string
  2: string
  3: number
}

export default function SemanticSearch() {
  const [searchQuery, setSearchQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const { showSnackBar } = useSnackBar()

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      showSnackBar('Please enter a search query', 'warning')
      return
    }

    setLoading(true)
    try {
      const response = await moduleService.semantic_search(searchQuery)
      setResults(response)
      if (response.length === 0) {
        showSnackBar('No results found', 'info')
      }
    } catch (error) {
      showSnackBar('Error performing semantic search', 'error')
      console.error('Search error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Semantic Search
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
          <TextField
            fullWidth
            label="Search Query"
            variant="outlined"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={loading}
            sx={{ minWidth: '120px' }}
          >
            {loading ? 'Searching...' : 'Search'}
          </Button>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Context</TableCell>
                <TableCell>Response</TableCell>
                <TableCell>Similarity</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.map((row) => (
                <TableRow key={row[0]}>
                  <TableCell>{row[0]}</TableCell>
                  <TableCell>{row[1]}</TableCell>
                  <TableCell>{row[2]}</TableCell>
                  <TableCell>{1 - row[3]}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Container>
  )
}
