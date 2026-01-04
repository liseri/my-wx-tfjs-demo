import * as tf from '@tensorflow/tfjs-core';

export class MiniprogramIOHandler implements tf.io.IOHandler {
    constructor(private modelPath: string) { }

    async load(): Promise<tf.io.ModelArtifacts> {
        const app = getApp();
        let modelJson = app.globalData.modelJsonData;
        let bin1Data = app.globalData.bin1Data;
        let bin2Data = app.globalData.bin2Data;
        let bin3Data = app.globalData.bin3Data;
        
        const fs = wx.getFileSystemManager();
        
        // 如果globalData中没有modelJson，尝试手动读取
        if (!modelJson) {
            console.log('modelJsonData not found in globalData, trying to read directly...');
            const modelJsonPath = `${this.modelPath}/json/model.json`;
            const modelJsonContent = fs.readFileSync(modelJsonPath, 'utf8');
            modelJson = JSON.parse(modelJsonContent);
            // 存储到globalData以便后续使用
            app.globalData.modelJsonData = modelJson;
        }
        
        // 准备权重数据缓冲区
        const buffers: ArrayBuffer[] = [];
        const weightSpecs: tf.io.WeightsManifestEntry[] = [];
        
        // 如果globalData中有所有权重数据，直接使用
        if (bin1Data && bin2Data && bin3Data) {
            buffers.push(bin1Data, bin2Data, bin3Data);
        } else {
            // 否则手动读取权重文件
            console.log('Weight data not found in globalData, trying to read directly...');
            for (const group of modelJson.weightsManifest) {
                for (const path of group.paths) {
                    const weightPath = `${this.modelPath}/${path}`;
                    const weightBuffer = fs.readFileSync(weightPath);
                    buffers.push(weightBuffer);
                }
                weightSpecs.push(...group.weights);
            }
        }
        
        // 如果weightSpecs为空，从modelJson中获取
        if (weightSpecs.length === 0) {
            for (const group of modelJson.weightsManifest) {
                weightSpecs.push(...group.weights);
            }
        }

        // 合并所有权重数据  
        const weightData = this.concatenateArrayBuffers(buffers);

        return {
            modelTopology: modelJson.modelTopology,
            weightSpecs: weightSpecs,
            weightData: weightData,
            format: modelJson.format,           // 确保传递格式  
            signature: modelJson.signature,     // 确保传递签名  
            userDefinedMetadata: modelJson.userDefinedMetadata, // 确保传递元数据  
            generatedBy: modelJson.generatedBy,
            convertedBy: modelJson.convertedBy
        };
    }

    private concatenateArrayBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
        const totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
        const result = new ArrayBuffer(totalLength);
        const view = new Uint8Array(result);

        let offset = 0;
        for (const buffer of buffers) {
            view.set(new Uint8Array(buffer), offset);
            offset += buffer.byteLength;
        }

        return result;
    }
}