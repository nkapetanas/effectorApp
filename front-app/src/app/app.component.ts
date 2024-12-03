import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EffectorComponent } from './effector/effector.component';

@Component({
  selector: 'app-root',
  template: '<app-effector></app-effector>',
  standalone: true,
  imports: [CommonModule, EffectorComponent]
})
export class AppComponent {
  title = 'effector-ui';
}
