//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__

#ifdef NEURALNETWORKLAYER_EXPORTS
#define GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API __declspec(dllexport)
#else
#define GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Utility.NeuralNetworkLayer.lib")
#endif
#endif


#include"../Layer/NeuralNetwork/ILayerDLLManager.h"
#include"../Layer/NeuralNetwork/ILayerDataManager.h"
#include"../Layer/Connect/ILayerConnectData.h"

#include<boost/filesystem.hpp>


namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	/** レイヤーをマージさせる方法 */
	enum LayerMergeType
	{
		LYAERMERGETYPE_MAX,		// 最大に併せる
		LYAERMERGETYPE_MIN,		// 最小に併せる
		LYAERMERGETYPE_LAYER0,	// 先頭レイヤーに併せる

		LAYERMERGETYPE_COUNT
	};

	
	/** レイヤーDLL管理クラスの作成(CPU用) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const wchar_t i_libraryDirPath[]);
	/** レイヤーDLL管理クラスの作成(GPU用) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const wchar_t i_libraryDirPath[]);

	//====================================
	// レイヤーデータを作成
	//====================================
	/** 複合ニューラルネットワーク.
		@param layerDLLManager	レイヤーDLL管理クラス.
		@param	inputLayerCount	入力レイヤー数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, U32 inputLayerCount);

	/** 畳込みニューラルネットワークレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	filterSize			フィルタサイズ.
		@param	outputChannelCount	フィルタの個数.
		@param	stride				フィルタの移動量.
		@param	paddingSize			パディングサイズ. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateConvolutionLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputChannelCount, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize,
		const wchar_t i_szWeightData[] = L"Default", const wchar_t i_szInitializerID[] = L"glorot_uniform");

	/** 入力拡張畳込みニューラルネットワークレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	filterSize			フィルタサイズ.
		@param	outputChannelCount	フィルタの個数.
		@param	dilation			入力の拡張量
		@param	stride				フィルタの移動量.
		@param	paddingSize			パディングサイズ. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateDilatedConvolutionLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputChannelCount, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> dilation, Vector3D<S32> stride, Vector3D<S32> paddingSize,
		const wchar_t i_szWeightData[] = L"Default", const wchar_t i_szInitializerID[] = L"glorot_uniform");

	/** 全結合ニューラルネットワークレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	neuronCount			ニューロン数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateFullyConnectLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputBufferCount, U32 neuronCount,
		const wchar_t i_szWeightData[] = L"Default", const wchar_t i_szInitializerID[] = L"glorot_uniform");

	/** 自己組織化マップレイヤー */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateSOMLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputBufferCount,
		U32 dimensionCount=2, U32 resolutionCount=16,
		F32 initValueMin=0.0f, F32 initValueMax=1.0f);


	/** 活性化レイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	activationType		活性化種別. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const wchar_t activationType[]);

	/** ドロップアウトレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	rate				ドロップアウト率.(0.0〜1.0).(0.0＝ドロップアウトなし,1.0=全入力無視) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateDropoutLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		F32 rate);

	/** ガウスノイズレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	average				発生する乱数の平均値
		@param	variance			発生する乱数の分散 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateGaussianNoiseLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, F32 average, F32 variance);


	/** プーリングレイヤー.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	filterSize			プーリング幅.
		@param	stride				フィルタ移動量. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreatePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> filterSize, Vector3D<S32> stride);

	/** バッチ正規化レイヤー
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateBatchNormalizationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputChannelCount);

	/** バッチ正規化レイヤー(チャンネル区別なし)
		@param	layerDLLManager		レイヤーDLL管理クラス. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateBatchNormalizationAllLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** スケール正規化レイヤー */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateNormalizationScaleLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** 指数平滑正規化レイヤー.
		@param	入力チャンネル数
		@param	平滑化時間数
		@param	初期化時間数 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateExponentialNormalizationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 i_InputChannelCount, U32 i_ExponentialTime, U32 i_InitParameterTimes);


	/** 広域平均プーリングレイヤー
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateGlobalAveragePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** GANにおけるDiscriminatorの出力レイヤー
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationDiscriminatorLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);


	/** アップサンプリングレイヤー
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	upScale				拡張率.
		@param	paddingUseValue		拡張部分の穴埋めに隣接する値を使用するフラグ. (true=UpConvolution, false=TransposeConvolution) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateUpSamplingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> upScale, bool paddingUseValue);

	/** チャンネル抽出レイヤー. 入力されたレイヤーの特定チャンネルを抽出する. 入力/出力データ構造でX,Y,Zは同じサイズ.
		@param	startChannelNo	開始チャンネル番号.
		@param	channelCount	抽出チャンネル数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateChooseChannelLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 startChannelNo, U32 channelCount);

	/** XYZ抽出レイヤー. 入力されたレイヤーの特定XYZ区間を抽出する. 入力/出力データ構造でCHは同じサイズ.
		@param	startPosition	開始XYZ位置.
		@param	boxSize			抽出XYZ数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateChooseBoxLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> startPosition, Vector3D<S32> boxSize);

	/** 後方伝搬範囲制限レイヤー. 出力レイヤーの特定XYZ区間以外の後方伝搬を停止する. 入力/出力データ構造でCH,x,y,zは同じサイズ.
		@param	startPosition	開始XYZ位置.
		@param	boxSize			抽出XYZ数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateLimitBackPropagationBoxLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> startPosition, Vector3D<S32> boxSize);

	/** 出力データ構造変換レイヤー.
		@param	ch	CH数.
		@param	x	X軸.
		@param	y	Y軸.
		@param	z	Z軸. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 ch, U32 x, U32 y, U32 z);
	/** 出力データ構造変換レイヤー.
		@param	outputDataStruct 出力データ構造 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& outputDataStruct);

	/** X=0を中心にミラー化する*/
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeMirrorXLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** X=0を中心に平方化する. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeSquareCenterCrossLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** X=0を中心に平方化する. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeSquareZeroSideLeftTopLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Gravisbell::U32 x, Gravisbell::U32 y);

	/** 信号の配列から値へ変換 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateSignalArray2ValueLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::U32 resolution);

	/** 確率の配列から値へ変換 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateProbabilityArray2ValueLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance, Gravisbell::U32 resolution);

	/** 値を信号の配列へ変換 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateValue2SignalArrayLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Gravisbell::F32 inputMinValue, Gravisbell::F32 inputMaxValue, Gravisbell::U32 resolution);


	/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	inputDataCount		入力されるレイヤーの個数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);


	/** 入力結合レイヤー(加算). 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
		@param	layerDLLManager		レイヤーDLL管理クラス
		@param	i_mergeType			ch数をマージさせる方法. 
		@param	i_scale				出力信号に掛け合わせるスカラー値 */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeAddLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		LayerMergeType i_mergeType, F32 i_scale = 1.0f);


	/** 入力結合レイヤー(平均). 入力されたレイヤーの値の平均をとる. 入力データ構造はX,Y,Zで同じサイズである必要がある.
		@param	layerDLLManager		レイヤーDLL管理クラス
		@param	i_mergeType			ch数をマージさせる方法. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeAverageLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		LayerMergeType i_mergeType);


	/** 入力結合レイヤー(最大値). 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
		@param	layerDLLManager		レイヤーDLL管理クラス
		@param	i_mergeType			ch数をマージさせる方法. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeMaxLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		LayerMergeType i_mergeType);


	/** 入力結合レイヤー(乗算). 入力されたレイヤーの値を乗算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
		@param	layerDLLManager		レイヤーDLL管理クラス
		@param	i_mergeType			ch数をマージさせる方法. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeMultiplyLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		LayerMergeType i_mergeType);


	/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる.
		@param	layerDLLManager		レイヤーDLL管理クラス.
		@param	inputDataStruct		入力データ構造.
		@param	inputDataCount		入力されるレイヤーの個数. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);



	/** レイヤーをネットワークの末尾に追加する.GUIDは自動割り当て.入力データ構造、最終GUIDも更新する. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix);

	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
		const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount);

	
	inline Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
		const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
	{
		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddLayer, onLayerFix, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
	}

	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix,
		std::vector<Gravisbell::GUID>& lpInputLayerGUID,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddlayer, onLayerFix, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}
	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		std::vector<Gravisbell::GUID> lpInputLayerGUID;
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddlayer, onLayerFix, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}


	/** ニューラルネットワークをバイナリファイルに保存する */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const wchar_t i_filePath[]);
	/** ニューラルネットワークをバイナリファイルから読み込むする */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const wchar_t i_filePath[]);


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif

