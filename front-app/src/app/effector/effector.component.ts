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

interface EffectorResponse {
  status: string;
  results: {
    pdp_plot?: string;
    rhale_plot?: string;
    [key: string]: string | undefined;
  };
}

interface AnalysisError {
  status: string;
  message: string;
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
  dataAcceptTypes = '.json,.npy';
  modelAcceptTypes = '.h5,.keras';
  config = {
    dataUrl: '',
    modelUrl: ''
  };

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
      }
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
    try {
      const formData = new FormData();

      if (this.dataSource === 'file' && this.selectedFile) {
        formData.append('data', this.selectedFile.file);
        formData.append('data_type', this.selectedFile.type);
      } else {
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

      const response = await this.http.post<EffectorResponse>(
        'http://localhost:5000/analyze',
        formData,
        {
          headers: new HttpHeaders({
            // Don't set Content-Type here as it's automatically set for FormData
            'Accept': 'application/json'
          }),
          withCredentials: true // Important for CORS with credentials
        }
      ).toPromise();

      if (response && response.status === 'success') {
        const plotKey = `${this.selectedMethod}_plot`;
        if (response.results[plotKey]) {
          this.plotImage = `data:image/png;base64,${response.results[plotKey]}`;
        }
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      this.loading = false;
    }
  }
}
