//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#include"stdafx.h"

#include"Utility/NeuralNetworkMaker.h"

#include<vector>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	class NeuralNetworkMaker : public INeuralNetworkMaker
	{
	private:
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager;	/**< DLL管理クラス */
		Layer::NeuralNetwork::ILayerDataManager& layerDataManager;	/**< レイヤー管理クラス */

		Layer::Connect::ILayerConnectData* pLayerConnectData;		/**< レイヤー接続情報 */
		bool onGetConnectData;

		std::vector<IODataStruct> lpInputDataStruct;

	public:
		/** コンストラクタ */
		NeuralNetworkMaker(const Layer::NeuralNetwork::ILayerDLLManager& i_layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& i_layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount)
			:	layerDLLManager		(i_layerDLLManager)
			,	layerDataManager	(i_layerDataManager)
			,	pLayerConnectData	(CreateNeuralNetwork(layerDLLManager,layerDataManager))
			,	onGetConnectData	(false)
		{
			for(U32 i=0; i<i_inputCount; i++)
				this->lpInputDataStruct.push_back(i_lpInputDataStruct[i]);
		}
		/** デストラクタ */
		virtual ~NeuralNetworkMaker()
		{
			if(!onGetConnectData)
				delete pLayerConnectData;
		}

		/** 作成したニューラルネットワークを取得する */
		Layer::Connect::ILayerConnectData* GetNeuralNetworkLayer()
		{
			this->onGetConnectData = true;

			return this->pLayerConnectData;
		}

	private:
		IODataStruct GetInputDataStruct(Gravisbell::GUID i_layerGUID)
		{
			return this->pLayerConnectData->GetOutputDataStruct(i_layerGUID, &this->lpInputDataStruct[0], (U32)this->lpInputDataStruct.size());;
		}

	public:
		//==================================
		// 基本レイヤー
		//==================================
		/** 畳込みニューラルネットワークレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_filterSize			フィルタサイズ.
			@param	i_outputChannelCount	フィルタの個数.
			@param	i_stride				フィルタの移動量.
			@param	i_paddingSize			パディングサイズ. */
		Gravisbell::GUID AddConvolutionLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, this->GetInputDataStruct(i_inputLayerGUID).ch, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize));
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** 全結合ニューラルネットワークレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_neuronCount			ニューロン数. */
		Gravisbell::GUID AddFullyConnectLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_neuronCount)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateFullyConnectLayer(layerDLLManager, layerDataManager, this->GetInputDataStruct(i_inputLayerGUID).GetDataCount(), i_neuronCount) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** 活性化レイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_activationType		活性化種別. */
		Gravisbell::GUID AddActivationLayer(const Gravisbell::GUID& i_inputLayerGUID, const wchar_t activationType[])
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateActivationLayer(layerDLLManager, layerDataManager, activationType) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** ドロップアウトレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_rate				ドロップアウト率.(0.0～1.0).(0.0＝ドロップアウトなし,1.0=全入力無視) */
		Gravisbell::GUID AddDropoutLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_rate)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateDropoutLayer(layerDLLManager, layerDataManager, i_rate) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** ガウスノイズレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_average			発生する乱数の平均値
			@param	i_variance			発生する乱数の分散 */
		Gravisbell::GUID AddGaussianNoiseLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_average, F32 i_variance)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateGaussianNoiseLayer(layerDLLManager, layerDataManager, i_average, i_variance) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** プーリングレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_filterSize			プーリング幅.
			@param	i_stride				フィルタ移動量. */
		Gravisbell::GUID AddPoolingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, Vector3D<S32> i_stride)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreatePoolingLayer(layerDLLManager, layerDataManager, i_filterSize, i_stride) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** バッチ正規化レイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		Gravisbell::GUID AddBatchNormalizationLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, this->GetInputDataStruct(i_inputLayerGUID).ch) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** バッチ正規化レイヤー(チャンネル区別なし)
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		Gravisbell::GUID AddBatchNormalizationAllLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateBatchNormalizationAllLayer(layerDLLManager, layerDataManager) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** スケール正規化レイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		Gravisbell::GUID AddNormalizationScaleLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateNormalizationScaleLayer(layerDLLManager, layerDataManager) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** 広域平均プーリングレイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		Gravisbell::GUID AddGlobalAveragePoolingLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateGlobalAveragePoolingLayer(layerDLLManager, layerDataManager) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** GANにおけるDiscriminatorの出力レイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		Gravisbell::GUID AddActivationDiscriminatorLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateActivationDiscriminatorLayer(layerDLLManager, layerDataManager) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** アップサンプリングレイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_upScale				拡張率.
			@param	i_paddingUseValue		拡張部分の穴埋めに隣接する値を使用するフラグ. (true=UpConvolution, false=TransposeConvolution) */
		Gravisbell::GUID AddUpSamplingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_upScale, bool i_paddingUseValue)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateUpSamplingLayer(layerDLLManager, layerDataManager, i_upScale, i_paddingUseValue) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** チャンネル抽出レイヤー. 入力されたレイヤーの特定チャンネルを抽出する. 入力/出力データ構造でX,Y,Zは同じサイズ.
			@param	i_inputLayerGUID	追加レイヤーの入力先レイヤーのGUID.
			@param	i_startChannelNo	開始チャンネル番号.
			@param	i_channelCount		抽出チャンネル数. */
		Gravisbell::GUID AddChooseChannelLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_startChannelNo, U32 i_channelCount)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateChooseChannelLayer(layerDLLManager, layerDataManager, i_startChannelNo, i_channelCount) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

		/** 出力データ構造変換レイヤー.
			@param	ch	CH数.
			@param	x	X軸.
			@param	y	Y軸.
			@param	z	Z軸. */
		Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 ch, U32 x, U32 y, U32 z)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateReshapeLayer(layerDLLManager, layerDataManager, ch, x, y, z) );
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}
		/** 出力データ構造変換レイヤー.
			@param	outputDataStruct 出力データ構造 */
		Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, const IODataStruct& outputDataStruct)
		{
			return AddReshapeLayer(i_inputLayerGUID, outputDataStruct.ch, outputDataStruct.x, outputDataStruct.y, outputDataStruct.z);
		}

	protected:
		/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeInputLayer(layerDLLManager, layerDataManager),
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる. */
		Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateResidualLayer(layerDLLManager, layerDataManager),
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		
	public:
		//==================================
		// 特殊レイヤー
		//==================================
		/** ニューラルネットワークにConvolution, Activationを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_CA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[])
		{
			GUID layerGUID = i_inputLayerGUID;

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** ニューラルネットワークにConvolution, BatchNormalization, Activationを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[])
		{
			GUID layerGUID = i_inputLayerGUID;

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** ニューラルネットワークにConvolution, BatchNormalization, Noise, Activationを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBNA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_noiseVariance)
		{
			GUID layerGUID = i_inputLayerGUID;

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ノイズ
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** ニューラルネットワークにBatchNormalization, Activation, Convolutionを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_BAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[])
		{
			GUID layerGUID = i_inputLayerGUID;

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			return layerGUID;
		}

		/** ニューラルネットワークにBatchNormalization, Activation. Fully-Connectを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_BAF(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[])
		{
			GUID layerGUID = i_inputLayerGUID;

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// 全結合
			layerGUID = this->AddFullyConnectLayer(layerGUID, i_outputChannelCount);

			return layerGUID;
		}

		/** ニューラルネットワークにFully-Connect, Activationを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_FA(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[])
		{
			return AddNeuralNetworkLayer_FAD(i_inputLayerGUID, i_outputChannelCount, i_activationType, 0.0f);
		}

		/** ニューラルネットワークにFully-Connect, Activation, Dropoutを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_FAD(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], F32 i_dropOutRate)
		{
			GUID layerGUID = i_inputLayerGUID;

			// 全結合
			layerGUID = this->AddFullyConnectLayer(layerGUID, i_outputChannelCount);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ドロップアウト
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** ニューラルネットワークにBatchNormalization, Noise, Activation, Convolutionを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_BNAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance)
		{
			GUID layerGUID = i_inputLayerGUID;

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ノイズ
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			return layerGUID;
		}

		/** ニューラルネットワークにNoise, BatchNormalization, Activation, Convolutionを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_NBAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance)
		{
			GUID layerGUID = i_inputLayerGUID;

			// ノイズ
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			return layerGUID;
		}

		/** ニューラルネットワークにConvolution, Activation, DropOutを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_CAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate)
		{
			GUID layerGUID = i_inputLayerGUID;

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ドロップアウト
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** ニューラルネットワークにConvolution, BatchNormalization, Activation, DropOutを行うレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate)
		{
			GUID layerGUID = i_inputLayerGUID;

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize);

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ドロップアウト
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** ニューラルネットワークにResNetレイヤーを追加する.(ノイズ付き) */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNet(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance)
		{
			GUID bypassLayerGUID = i_inputLayerGUID;
			GUID layerGUID = i_inputLayerGUID;

			U32 outputChannel = this->GetInputDataStruct(i_inputLayerGUID).ch;

			for(U32 layerNum=0; layerNum<i_layerCount-1; layerNum++)
			{
				// 2層目
				layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, outputChannel, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU");
			}

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ノイズ
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, L"ReLU");

			// ドロップアウト
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, outputChannel, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2) );

			// Residual
			layerGUID = INeuralNetworkMaker::AddResidualLayer(layerGUID, bypassLayerGUID);


			return S_OK;
		}

		/** ニューラルネットワークにResNetレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_front_layerCount, U32 i_back_layerCount, F32 i_noiseVariance)
		{
			GUID bypassLayerGUID = i_inputLayerGUID;
			GUID layerGUID = i_inputLayerGUID;

			U32 inputChannelCount = this->GetInputDataStruct(i_inputLayerGUID).ch;

			// 前半
			for(S32 layerNum=0; layerNum<(S32)i_front_layerCount-1; layerNum++)
			{
				layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, inputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU");
			}

			// CH数を変更
			layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU");

			// 後半
			for(S32 layerNum=0; layerNum<(S32)i_back_layerCount-1; layerNum++)
			{
				// 2層目
				layerGUID = AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU");
			}

			// バッチ正規化
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ノイズ
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// 活性化
			layerGUID = this->AddActivationLayer(layerGUID, L"ReLU");

			// ドロップアウト
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			// 畳み込み
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2) );

			// Residual
			layerGUID = INeuralNetworkMaker::AddResidualLayer(layerGUID, bypassLayerGUID);


			return S_OK;
		}

		/** ニューラルネットワークにResNetレイヤーを追加する. */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize_single(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance)
		{
			return this->AddNeuralNetworkLayer_ResNetResize(i_inputLayerGUID, i_filterSize, i_outputChannelCount, i_dropOutRate, 1, max(0, (Gravisbell::S32)i_layerCount-1), i_noiseVariance);
		}
	};

	/** ニューラルネットワーク作成クラスを取得する */
	INeuralNetworkMaker* CreateNeuralNetworkManaker(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount)
	{
		return new NeuralNetworkMaker(layerDLLManager, layerDataManager, i_lpInputDataStruct, i_inputCount);
	}

}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
