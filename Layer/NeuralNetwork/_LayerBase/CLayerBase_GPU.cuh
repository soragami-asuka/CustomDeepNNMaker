//============================================
// 全てのニューラルネットワーク系レイヤーのベースとなる共通処理
// 自前で実装可能であれば必ず継承する必要はない
//============================================
#ifndef __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_GPU_H__
#define __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_GPU_H__


#include"CLayerBase.h"

#include<Common/Common.h>
#include<Common/ErrorCode.h>
#include<Common/ITemporaryMemoryManager.h>

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	//=================================
	// 単一入力 / 単一出力
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2SingleLayerBase_GPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<F32>				lpInputBuffer_h;		/**< 入力バッファ <バッチ数><入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** コンストラクタ */
		CNNSingle2SingleLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2SingleLayerBase_GPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
			// 一時バッファのサイズを登録
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]",   sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"dinput[0]",  sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());

			// 出力バッファを確保
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// 事前演算
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2SingleLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2SingleLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// 演算処理
		//================================
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
		{
			// 入力バッファをコピー
			if(this->lpInputBuffer_h.empty())
				this->lpInputBuffer_h.resize(this->GetInputBufferCount() * this->GetBatchSize());
			memcpy(&this->lpInputBuffer_h[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer_h.size());

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpy(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファを確保
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device(lppInputBuffer, lppOutputBuffer);

			// バッファを開放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// 出力バッファをホストにコピー
			cudaMemcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);


			return err;
		}


		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyHostToHost);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// 学習処理
		//================================
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// 出力誤差バッファをデバイスにコピー
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力誤差バッファを確保
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// 同期
			cudaThreadSynchronize();

			// 演算
			ErrorCode err = CalculateDInput_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// 入力誤差バッファをデバイスにコピー
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// バッファ解放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// 出力誤差バッファをデバイスにコピー
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力誤差バッファを確保
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// 同期
			cudaThreadSynchronize();

			// 演算
			ErrorCode err = Training_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// 入力誤差バッファをデバイスにコピー
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// バッファ解放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}
	};

	//=================================
	// 単一入力 / 複数出力
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2MultLayerBase_GPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<F32>				lpInputBuffer_h;		/**< 入力バッファ <バッチ数><入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		std::vector<BATCH_BUFFER_POINTER>	lppDOutputBuffer_d;	/**< 出力誤差バッファの配列(デバイスメモリ). <出力先レイヤー数> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

		std::vector<std::wstring>		lpDOutputBufferID;		/**< 出力誤差バッファに付けられたID */

	public:
		/** コンストラクタ */
		CNNSingle2MultLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2MultLayerBase_GPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
			// IDリストを初期化
			this->lpDOutputBufferID.clear();

			// 一時バッファのサイズを登録
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]",   sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"dinput[0]",  sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				wchar_t szID[256];
				swprintf(szID, sizeof(szID)-1, L"doutput[%d]", outputNo);

				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID, sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
				this->lpDOutputBufferID.push_back(szID);
			}

			// 出力バッファを確保
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			// 出力誤差バッファ配列を確保
			this->lppDOutputBuffer_d.resize(this->GetOutputToLayerCount());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// 事前演算
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2MultLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2MultLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// 演算処理
		//================================
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
		{
			// 入力バッファをコピー
			if(this->lpInputBuffer_h.empty())
				this->lpInputBuffer_h.resize(this->GetInputBufferCount() * this->GetBatchSize());
			memcpy(&this->lpInputBuffer_h[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer_h.size());

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpy(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファを確保
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device(lppInputBuffer, lppOutputBuffer);

			// バッファを開放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// 出力バッファをデバイスにコピー
			cudaMemcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);


			return err;
		}


		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyHostToHost);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// 学習処理
		//================================
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			// 出力誤差バッファ配列を作成
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				// 出力誤差バッファのアドレスを配列に格納
				this->lppDOutputBuffer_d[outputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

				// 出力誤差バッファをデバイスにコピー
				cudaMemcpyAsync(this->lppDOutputBuffer_d[outputNo], i_lppDOutputBuffer[outputNo], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力誤差バッファを確保
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// 同期
			cudaThreadSynchronize();

			// 演算
			ErrorCode err = CalculateDInput_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, (const float**)&this->lppDOutputBuffer_d[0]);


			// 入力誤差バッファをデバイスにコピー
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// バッファ解放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

			return err;
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			// 出力誤差バッファ配列を作成
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				// 出力誤差バッファのアドレスを配列に格納
				this->lppDOutputBuffer_d[outputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

				// 出力誤差バッファをデバイスにコピー
				cudaMemcpyAsync(this->lppDOutputBuffer_d[outputNo], i_lppDOutputBuffer[outputNo], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, &this->lpInputBuffer_h[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力誤差バッファを確保
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// 同期
			cudaThreadSynchronize();

			// 演算
			ErrorCode err = Training_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, (const float**)&this->lppDOutputBuffer_d[0]);


			// 入力誤差バッファをデバイスにコピー
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// バッファ解放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

			return err;
		}
	};

	//=================================
	// 複数入力 / 単一出力
	//=================================
	template<class Layer, class LayerData>
	class CNNMult2SingleLayerBase_GPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<std::vector<F32>>	lppInputBuffer_h;		/**< 入力バッファ <入力レイヤー数><バッチ数><入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		std::vector<std::wstring>		lpInputBufferID;		/**< 入力バッファに付けられたID */
		std::vector<std::wstring>		lpDInputBufferID;		/**< 入力誤差バッファに付けられたID */

		std::vector<BATCH_BUFFER_POINTER>	lppInputBuffer_d;	/**< 入力バッファの配列(デバイスメモリ). <入力元レイヤー数> */
		std::vector<BATCH_BUFFER_POINTER>	lppDInputBuffer_d;	/**< 入力誤差バッファの配列(デバイスメモリ). <入力元レイヤー数> */


		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** コンストラクタ */
		CNNMult2SingleLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_lpInputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNMult2SingleLayerBase_GPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
			this->lpInputBufferID.clear();
			this->lpDInputBufferID.clear();

			// 一時バッファのサイズを登録
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				wchar_t szID[256];

				swprintf(szID, sizeof(szID)-1, L"input[%d]", inputNo);
				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID,  sizeof(F32) * this->GetInputBufferCount(inputNo) * this->GetBatchSize());
				this->lpInputBufferID.push_back(szID);

				swprintf(szID, sizeof(szID)-1, L"dinput[%d]", inputNo);
				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID,  sizeof(F32) * this->GetInputBufferCount(inputNo) * this->GetBatchSize());
				this->lpDInputBufferID.push_back(szID);
			}
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());

			// 出力バッファを確保
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			// 入力/入力誤差バッファのアドレス配列を作成
			this->lppInputBuffer_d.resize(this->GetInputDataCount());
			this->lppDInputBuffer_d.resize(this->GetInputDataCount());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// 事前演算
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNMult2SingleLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNMult2SingleLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// 演算処理
		//================================
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[])
		{
			// 入力バッファをコピー
			if(this->lppInputBuffer_h.empty())
			{
				this->lppInputBuffer_h.resize(this->GetInputDataCount());

				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
					this->lppInputBuffer_h[inputNo].resize(this->GetInputBufferCount(inputNo) * this->GetBatchSize());
			}
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				memcpy(&this->lppInputBuffer_h[inputNo][0], i_lppInputBuffer[inputNo], sizeof(F32)*this->lppInputBuffer_h[inputNo].size());

			// 入力バッファをデバイスにコピー
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpy(this->lppInputBuffer_d[inputNo], i_lppInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// 出力バッファを確保
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device((const F32**)&this->lppInputBuffer_d[0], lppOutputBuffer);

			// バッファを開放
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

			// 出力バッファをデバイスにコピー
			cudaMemcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);


			return err;
		}


		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyHostToHost);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// 学習処理
		//================================
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// 出力誤差バッファをデバイスにコピー
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpyAsync(this->lppInputBuffer_d[inputNo], &this->lppInputBuffer_h[inputNo][0], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// 入力誤差バッファを確保
			F32** lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					this->lppDInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
				lppDInputBuffer = &this->lppDInputBuffer_d[0];
			}

			// 同期
			cudaThreadSynchronize();

			// 演算
			ErrorCode err = CalculateDInput_device((const F32**)&this->lppInputBuffer_d[0], lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// 入力誤差バッファをホストにコピー
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					cudaMemcpy(o_lppDInputBuffer[inputNo], lppDInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
					this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
			}

			// バッファ解放
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// 出力誤差バッファをデバイスにコピー
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 出力バッファをデバイスにコピー
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, &this->lpOutputBuffer_h[0], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// 入力バッファをデバイスにコピー
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpyAsync(this->lppInputBuffer_d[inputNo], &this->lppInputBuffer_h[inputNo][0], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// 入力誤差バッファを確保
			F32** lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					this->lppDInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
				lppDInputBuffer = &this->lppDInputBuffer_d[0];
			}

			// 同期
			cudaThreadSynchronize();


			// 演算
			ErrorCode err = Training_device((const F32**)&this->lppInputBuffer_d[0], lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// 入力誤差バッファをホストにコピー
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					cudaMemcpy(o_lppDInputBuffer[inputNo], lppDInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
					this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
			}

			// バッファ解放
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
