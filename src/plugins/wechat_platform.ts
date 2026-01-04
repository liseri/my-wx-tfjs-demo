/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tfjs from '@tensorflow/tfjs-core';
// import * as cpu from '@tensorflow/tfjs-backend-cpu';
import * as webgl_backend from '@tensorflow/tfjs-backend-webgl';
// import * as wasm from '@tensorflow/tfjs-backend-wasm';
// import * as wasm from '@tensorflow/tfjs-backend-webgpu';
import { atob, btoa } from 'abab';
// @ts-ignore: Cannot find module 'text-encoder' or its corresponding type declarations
import { TextDecoder, TextEncoder } from 'text-encoder';

export interface SystemConfig {
  /**
   * A function used to override the `window.fetch` function.
   */
  fetchFunc: Function;
  
  /**
   * The WeChat offline canvas, can be created by calling
   * wx.createOfflineCanvas()
   */
  // tslint:disable-next-line:no-any
  canvas: any;
  /**
   * Optional name of backend to use
   */
  // tslint:disable-next-line:no-any
  backendName?: string;
}

export let systemFetchFunc: Function;

// Implement the WeChat Platform for TFJS
export class MiniprogramPlatform implements tfjs.Platform {
  constructor(fetchFunc: Function) {
    systemFetchFunc = fetchFunc;
  }
  fetch(path: string, requestInits?: any): Promise<any> {
    return systemFetchFunc(path, requestInits);
  }
  now(): number {
    return Date.now();
  }
  encode(text: string, encoding: string): Uint8Array {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
        `Browser's encoder only supports utf-8, but got ${encoding}`);
    }
    return new TextEncoder(encoding).encode(text);
  }
  decode(bytes: Uint8Array, encoding: string): string {
    return new TextDecoder(encoding).decode(bytes);
  }
  isTypedArray(a: unknown): a is Float32Array | Int32Array | Uint8Array | Uint8ClampedArray {
    return ArrayBuffer.isView(a) && !(a instanceof DataView);
  }
}

/**
 * Setup the fetch polyfill and backend for WeChat.
 * @param config: SystemConfig object contains Tensorflow.js runtime, fetch
 *     polyfill and WeChat offline canvas.
 * @param debug: flag to enable/disable debugging.
 */
export async function setupWechatPlatform(config: SystemConfig, debug = false) {
  // 优先使用config中指定的backendName，如果没有则自动检测
  let backendName = config.backendName;
  
  if (debug) {
    console.log("//==tf getBackend=", tfjs.getBackend);
  }
  tfjs.ENV.setPlatform('miniprogram', new MiniprogramPlatform(config.fetchFunc));
  setBase64Methods(tfjs);
  
  // 定义后端优先级顺序：WebGL > CPU
  // 注意：微信小程序环境下，WebGPU和WASM的支持有限，暂不考虑
  const backendPriority = backendName ? [backendName] : ['webgl', 'cpu'];
  
  let selectedBackend = null;
  
  for (const candidateBackend of backendPriority) {
    try {
      if (debug) {
        console.log(`Trying to initialize backend: ${candidateBackend}`);
      }
      
      if (candidateBackend === 'webgl') {
        if (config.canvas) {
          await initWebGL(tfjs, webgl_backend, config.canvas, candidateBackend, debug);
          selectedBackend = candidateBackend;
          break;
        } else {
          if (debug) {
            console.log('WebGL backend requires a canvas, skipping');
          }
          continue;
        }
      } else if (candidateBackend === 'cpu') {
        // CPU backend setup
        await tfjs.setBackend('cpu');
        selectedBackend = candidateBackend;
        break;
      } 
      // 其他后端（如wasm、webgpu）暂不支持微信小程序环境
    } catch (error) {
      if (debug) {
        console.log(`Failed to initialize backend ${candidateBackend}:`, error);
      }
      // 继续尝试下一个后端
      continue;
    }
  }
  
  if (selectedBackend === null) {
    // 所有后端都失败，默认使用CPU
    if (debug) {
      console.log('All backends failed, falling back to CPU backend');
    }
    await tfjs.setBackend('cpu');
  }
  
  if (debug) {
    console.log('Final backend = ', tfjs.getBackend());
  }
}

/**
 * Polyfill btoa and atob method on the global scope which will be used by
 * model parser.
 */
export function setBase64Methods(tf: typeof tfjs) {
  tf.ENV.global.btoa = btoa;
  tf.ENV.global.atob = atob;
}
/**
 * Initialize webgl backend using the WebGLRenderingContext from the webgl
 * canvas node.
 * @param canvas: webgl canvas node container return from node selector.
 * @param platform: platform name where the mini app is running (ios, android,
 *     devtool).
 * @param debug: enable/disable debug logging.
 */
const BACKEND_PRIORITY = 2;
export async function initWebGL(
  // tslint:disable-next-line:no-any
  tf: typeof tfjs, webgl: typeof webgl_backend, canvas: any,
  backendName, debug = false) {
  if (tf.findBackend(backendName) == null) {
    const WEBGL_ATTRIBUTES = {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true
    };
    const gl = canvas.getContext('webgl', WEBGL_ATTRIBUTES);
    if (debug) {
      console.log('start backend registration');
    }
    try {
      tf.registerBackend(backendName, () => {
        webgl.setWebGLContext(1, gl);
        tf.ENV.set('WEBGL_VERSION', 1);
        const context = new webgl.GPGPUContext(gl);
        return new webgl.MathBackendWebGL(context);
      }, BACKEND_PRIORITY);

      // Register all the webgl kernels on the rn-webgl backend
      const kernels = tf.getKernelsForBackend('webgl');
      kernels.forEach(kernelConfig => {
        const newKernelConfig = Object.assign({}, kernelConfig, { backendName });
        tf.registerKernel(newKernelConfig);
      });
    } catch (e: any) {
      throw (new Error(`Failed to register Webgl backend: ${e.message || e}`));
    }
  }
  await tf.setBackend(backendName);
  if (debug) {
    console.log('current backend = ', tf.getBackend());
  }
}
