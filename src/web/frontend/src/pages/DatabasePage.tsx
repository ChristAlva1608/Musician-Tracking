import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Pagination,
  Chip,
  SelectChangeEvent
} from '@mui/material';
import { API_BASE_URL } from '../config';

interface TableInfo {
  name: string;
  description: string;
  type: string;
}

interface TableData {
  table_name: string;
  data: any[];
  limit: number;
  offset: number;
  total_count: number;
}

export const DatabasePage: React.FC = () => {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [tableData, setTableData] = useState<TableData | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<number>(1);
  const [rowsPerPage] = useState<number>(50);

  // Fetch available tables on mount
  useEffect(() => {
    fetchTables();
  }, []);

  // Fetch table data when table is selected or page changes
  useEffect(() => {
    if (selectedTable) {
      fetchTableData();
    }
  }, [selectedTable, page]);

  const fetchTables = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/database/tables`);
      if (!response.ok) throw new Error('Failed to fetch tables');
      const data = await response.json();
      setTables(data.tables);

      // Auto-select first table if available
      if (data.tables.length > 0 && !selectedTable) {
        setSelectedTable(data.tables[0].name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tables');
    } finally {
      setLoading(false);
    }
  };

  const fetchTableData = async () => {
    if (!selectedTable) return;

    try {
      setLoading(true);
      setError(null);

      const offset = (page - 1) * rowsPerPage;
      const response = await fetch(
        `${API_BASE_URL}/api/database/table/${selectedTable}/data?limit=${rowsPerPage}&offset=${offset}`
      );

      if (!response.ok) throw new Error('Failed to fetch table data');

      const data: TableData = await response.json();
      setTableData(data);

      // Extract columns from first row of data
      if (data.data && data.data.length > 0) {
        setColumns(Object.keys(data.data[0]));
      } else {
        // Fetch column info even if no data
        fetchTableColumns();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch table data');
    } finally {
      setLoading(false);
    }
  };

  const fetchTableColumns = async () => {
    if (!selectedTable) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/database/table/${selectedTable}/columns`
      );

      if (!response.ok) throw new Error('Failed to fetch columns');

      const data = await response.json();
      setColumns(data.columns);
    } catch (err) {
      console.error('Failed to fetch columns:', err);
    }
  };

  const handleTableChange = (event: SelectChangeEvent<string>) => {
    setSelectedTable(event.target.value);
    setPage(1); // Reset to first page
    setTableData(null); // Clear previous data
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
  };

  const formatCellValue = (value: any): string => {
    if (value === null) return 'null';
    if (value === undefined) return '';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  const getTableTypeColor = (type: string) => {
    switch (type) {
      case 'detection': return 'primary';
      case 'alignment': return 'secondary';
      case 'chunk_alignment': return 'success';
      case 'transcript': return 'info';
      default: return 'default';
    }
  };

  const totalPages = tableData ? Math.ceil(tableData.total_count / rowsPerPage) : 0;

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Database Viewer
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Browse and query Supabase database tables
        </Typography>
      </Box>

      {/* Table Selector */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <FormControl sx={{ minWidth: 300 }}>
          <InputLabel id="table-select-label">Select Table</InputLabel>
          <Select
            labelId="table-select-label"
            id="table-select"
            value={selectedTable}
            label="Select Table"
            onChange={handleTableChange}
            disabled={loading}
          >
            {tables.map((table) => (
              <MenuItem key={table.name} value={table.name}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>{table.name}</span>
                  <Chip
                    label={table.type}
                    size="small"
                    color={getTableTypeColor(table.type) as any}
                  />
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {selectedTable && tableData && (
          <Typography variant="body2" color="text.secondary">
            Showing {Math.min(rowsPerPage, tableData.data.length)} of {tableData.total_count} records
          </Typography>
        )}
      </Box>

      {/* Table Description */}
      {selectedTable && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            {tables.find(t => t.name === selectedTable)?.description}
          </Typography>
        </Box>
      )}

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Loading Spinner */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Data Table */}
      {!loading && tableData && tableData.data.length > 0 && (
        <>
          <TableContainer component={Paper} sx={{ mb: 2, maxHeight: 600 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  {columns.map((column) => (
                    <TableCell
                      key={column}
                      sx={{
                        fontWeight: 'bold',
                        backgroundColor: 'background.paper',
                        whiteSpace: 'nowrap'
                      }}
                    >
                      {column}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableData.data.map((row, index) => (
                  <TableRow
                    key={index}
                    sx={{ '&:hover': { backgroundColor: 'action.hover' } }}
                  >
                    {columns.map((column) => (
                      <TableCell
                        key={column}
                        sx={{
                          maxWidth: 300,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                        title={formatCellValue(row[column])}
                      >
                        {formatCellValue(row[column])}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Pagination */}
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}

      {/* No Data Message */}
      {!loading && tableData && tableData.data.length === 0 && (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No data found in {selectedTable}
          </Typography>
        </Paper>
      )}

      {/* No Table Selected */}
      {!loading && !selectedTable && tables.length === 0 && (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No tables available or database not configured
          </Typography>
        </Paper>
      )}
    </Container>
  );
};