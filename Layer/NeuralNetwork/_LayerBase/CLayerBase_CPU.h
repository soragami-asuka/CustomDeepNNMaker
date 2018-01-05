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
			ホストメモリが渡される
			@param i_lppInputBuffer		入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@param o_lppOutputBuffer	出力データバッファ. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			return this->Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  Training_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}
	};


	//=================================
	// 単一入力 / 複数出力
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2MultLayerBase_CPU : public Layer
	{
	protected:
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
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			return this->Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			return  CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			return  Training_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}
	};


	//=================================
	// 複数入力 / 単一出力
	//=================================
	template<class Layer, class LayerData>
	class CNNMult2SingleLayerBase_CPU : public Layer
	{
	protected:
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
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			return this->Calculate_device(i_lppInputBuffer,o_lppOutputBuffer);
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer )
		{
			return  Training_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
