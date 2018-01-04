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


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	//=================================
	// 単一入力 / 単一出力
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2SingleLayerBase_CPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<F32>				lpInputBuffer_h;		/**< 入力バッファ <バッチ数><入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** コンストラクタ */
		CNNSingle2SingleLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2SingleLayerBase_CPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
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

			return this->Calculate_device(&this->lpInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// 出力バッファをレイヤー内メモリにコピー
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

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

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

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
			return  CalculateDInput_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  Training_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


	//=================================
	// 単一入力 / 複数出力
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2MultLayerBase_CPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<F32>				lpInputBuffer_h;		/**< 入力バッファ <バッチ数><入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** コンストラクタ */
		CNNSingle2MultLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNSingle2MultLayerBase_CPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
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

			return this->Calculate_device(&this->lpInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// 出力バッファをレイヤー内メモリにコピー
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

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

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

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
			return  CalculateDInput_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			return  Training_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


	//=================================
	// 複数入力 / 単一出力
	//=================================
	template<class Layer, class LayerData>
	class CNNMult2SingleLayerBase_CPU : public Layer
	{
	protected:
		// 入出力バッファ
		std::vector<std::vector<F32>>	lppInputBuffer_h;		/**< 入力バッファ <入力レイヤー数><バッチ数*入力信号数> */
		std::vector<F32>				lpOutputBuffer_h;		/**< 出力バッファ <バッチ数><出力信号数> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** コンストラクタ */
		CNNMult2SingleLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_lpInputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** デストラクタ */
		virtual ~CNNMult2SingleLayerBase_CPU()
		{
		}

	public:
		/** 入出力バッファの確保と一時バッファの予約 */
		ErrorCode ReserveMemory()
		{
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

			return this->Calculate_device((const F32**)&this->lppInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// 演算
			ErrorCode err = Layer::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

			// 出力バッファをレイヤー内メモリにコピー
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

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

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// 学習処理
		//================================
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  CalculateDInput_device((const F32**)&this->lppInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  Training_device((const F32**)&this->lppInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
