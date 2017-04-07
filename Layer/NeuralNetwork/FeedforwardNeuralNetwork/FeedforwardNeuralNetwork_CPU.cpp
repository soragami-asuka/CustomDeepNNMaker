//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// CPU処理
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_CPU : public FeedforwardNeuralNetwork_Base
	{
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_CPU(const ILayerDLLManager& layerDLLManager, const Gravisbell::GUID& i_guid)
			:	FeedforwardNeuralNetwork_Base(layerDLLManager, i_guid)
		{
		}
		/** コンストラクタ
			@param	i_inputGUID	入力信号に割り当てられたGUID.自分で作ることができないので外部で作成して引き渡す. */
		FeedforwardNeuralNetwork_CPU(const ILayerDLLManager& layerDLLManager, const Gravisbell::GUID& i_guid, const Gravisbell::GUID& i_inputLayerGUID)
			:	FeedforwardNeuralNetwork_Base(layerDLLManager, i_guid, i_inputLayerGUID)
		{
		}
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_CPU()
		{
		}

	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKind(void)const
		{
			return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_CPU;
		}

	public:
		//====================================
		// 入出力バッファ関連
		//====================================
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpOutputBuffer[batchNum], FeedforwardNeuralNetwork_Base::GetOutputBuffer()[batchNum], sizeof(F32)*outputBufferCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 inputBufferCount = this->GetInputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpDInputBuffer[batchNum], FeedforwardNeuralNetwork_Base::GetDInputBuffer()[batchNum], sizeof(F32)*inputBufferCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


/** Create a layer for CPU processing.
	* @param GUID of layer to create.
	*/
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerCPU(Gravisbell::GUID guid, const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager)
{
	if(pLayerDLLManager == NULL)
		return NULL;


	return new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_CPU(*pLayerDLLManager, guid);
}