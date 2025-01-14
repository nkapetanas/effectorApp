import {HttpClient, HttpClientModule, HttpErrorResponse, HttpHeaders} from '@angular/common/http';
import { DataService } from '../services/data.service';
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';

interface FileData {
  file: File;
  data?: any;
  type: string;
}

interface EffectorResults {
  [key: string]: string | undefined;
  pdp_plot?: string;
  rhale_plot?: string;
  regional_rhale_plot?: string;
  regional_pdp_plot?: string;
  partitioning_info?: string;
}

interface EffectorConfig {
  dataUrl: string;
  modelUrl: string;
  featureNames?: string[];
  targetName: string;
  instances?: number;
  method: 'pdp' | 'rhale' | 'regional_rhale' | 'regional_pdp';
  nodeIdx?: number;  // For regional methods
  heterogeneity?: 'std' | 'ice';  // For RHALE and RegionalPDP
}

interface EffectorResponse {
  status: string;
  results: {
    pdp_plot?: string;
    rhale_plot?: string;
    regional_rhale_plot?: string;
    regional_pdp_plot?: string;
    partitioning_info?: string;  // For regional methods
  };
}


@Component({
  selector: 'app-effector',
  templateUrl: './effector.component.html',
  styleUrls: ['./effector.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    MatCardModule,
    MatTabsModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatFormFieldModule
  ]
})
export class EffectorComponent {
  selectedMethod: string = 'pdp';
  featureIndex: number = 0;
  plotImage: string | null = null;
  loading: boolean = false;
  selectedFile: FileData | null = null;
  selectedModel: FileData | null = null;
  dataSource: string = 'url';
  dataAcceptTypes = '.json,.npy,.csv';
  modelAcceptTypes = '.h5,.keras,.pkl';
  config = {
    dataUrl: '',
    modelUrl: ''
  };

  // Added missing properties
  targetName: string = 'prediction';
  nodeIdx: number = 1;
  heterogeneity: 'std' | 'ice' = 'std';
  partitioningInfo: string | null = null;
  results: {
    pdp_plot?: string;
    rhale_plot?: string;
    regional_rhale_plot?: string;
    regional_pdp_plot?: string;
    partitioning_info?: string;
  } = {};

  constructor(
    private http: HttpClient,
    private dataService: DataService
  ) {}

  async handleDataFileChange(event: Event): Promise<void> {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
      if (file.name.endsWith('.json')) {
        const jsonData = await this.dataService.readFileAsJson(file);
        this.selectedFile = {
          file,
          data: jsonData,
          type: 'json'
        };
      } else if (file.name.endsWith('.npy')) {
        this.selectedFile = {
          file,
          type: 'npy'
        };
      } else if (file.name.endsWith('.csv')) {
        this.selectedFile = {
          file,
          type: 'csv'
        };
      }
      console.log('Selected file:', this.selectedFile);
    } catch (error) {
      console.error('Error reading file:', error);
    }
  }


  async handleModelFileChange(event: Event): Promise<void> {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    try {
      const modelBuffer = await this.dataService.readModelFile(file);
      this.selectedModel = {
        file,
        data: modelBuffer,
        type: file.name.endsWith('.h5') ? 'h5' : 'keras'
      };
    } catch (error) {
      console.error('Error reading model file:', error);
    }
  }

  clearModelFile(): void {
    this.selectedModel = null;
  }

  async analyze(): Promise<void> {
    this.loading = true;
    this.plotImage = null;  // Clear previous results
    this.partitioningInfo = null;

    try {
      const formData = new FormData();

      if (this.dataSource === 'file' && this.selectedFile) {
        console.log('Sending file:', this.selectedFile);
        formData.append('data', this.selectedFile.file);
        formData.append('data_type', this.selectedFile.type);
      } else {
        console.log('Sending URL:', this.config.dataUrl);
        formData.append('data_url', this.config.dataUrl);
      }

      if (this.selectedModel) {
        formData.append('model', this.selectedModel.file);
        formData.append('model_type', this.selectedModel.type);
      } else {
        formData.append('model_url', this.config.modelUrl);
      }

      formData.append('method', this.selectedMethod);
      formData.append('feature_index', this.featureIndex.toString());
      formData.append('target_name', this.targetName);
      formData.append('node_idx', this.nodeIdx.toString());
      formData.append('heterogeneity', this.heterogeneity);

      console.log('Sending request with method:', this.selectedMethod);

      const response = await this.http.post<EffectorResponse>(
        'http://localhost:5000/analyze',
        formData
      ).toPromise();

      console.log('Received response:', response);

      if (response && response.status === 'success' && response.results) {
        // Store all results
        this.results = response.results;

        // Get the plot key based on method
        const plotKey = `${this.selectedMethod}_plot` as keyof EffectorResponse['results'];
        const plotData = response.results[plotKey];

        if (plotData) {
          console.log(`Found plot data for ${plotKey}`);
          this.plotImage = `data:image/png;base64,${plotData}`;
        } else {
          console.warn(`No plot data found for ${plotKey}`);
        }

        // Handle partitioning info
        if (['regional_rhale', 'regional_pdp'].includes(this.selectedMethod)) {
          this.partitioningInfo = response.results.partitioning_info || null;
          if (this.partitioningInfo) {
            console.log('Received partitioning info');
          }
        }
      } else {
        console.error('Invalid response:', response);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      if (error instanceof HttpErrorResponse) {
        const errorMessage = error.error?.message || error.message;
        console.error('Server error:', errorMessage);
        // You might want to show this error to the user
        // Add error display in your template
      }
    } finally {
      this.loading = false;
    }
  }
}
