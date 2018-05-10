//==================================
// ニューラルネットワークのレイヤー用クラス
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORK_MAKER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORK_MAKER_H__

#include"NeuralNetworkLayer.h"

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	class INeuralNetworkMaker
	{
	public:
		/** コンストラクタ */
		INeuralNetworkMaker(){}
		/** デストラクタ */
		virtual ~INeuralNetworkMaker(){}

	public:
		/** 作成したニューラルネットワークを取得する */
		virtual Layer::Connect::ILayerConnectData* GetNeuralNetworkLayer()=0;

		/** 指定レイヤーの出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct(const Gravisbell::GUID& i_layerGUID) = 0;

	public:
		//==================================
		// レイヤー追加処理
		//==================================
		virtual Gravisbell::GUID AddLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Gravisbell::Layer::ILayerData* i_pLayerData, bool onLayerFix=false) = 0;

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
		virtual Gravisbell::GUID AddConvolutionLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize,
			const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** 入力拡張畳込みニューラルネットワークレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_filterSize			フィルタサイズ.
			@param	i_outputChannelCount	フィルタの個数.
			@param	i_dilation				入力の拡張量.
			@param	i_stride				フィルタの移動量.
			@param	i_paddingSize			パディングサイズ. */
		virtual Gravisbell::GUID AddDilatedConvolutionLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_dilation, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize,
			const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;


		/** 全結合ニューラルネットワークレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_neuronCount			ニューロン数. */
		virtual Gravisbell::GUID AddFullyConnectLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_neuronCount, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** 自己組織化マップレイヤー */
		virtual Gravisbell::GUID AddSOMLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 dimensionCount=2, U32 resolutionCount=16, F32 initValueMin=0.0f, F32 initValueMax=1.0f, bool onLayerFix=false) = 0;

		/** 活性化レイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_activationType		活性化種別. */
		virtual Gravisbell::GUID AddActivationLayer(const Gravisbell::GUID& i_inputLayerGUID, const wchar_t activationType[]) = 0;

		/** ドロップアウトレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_rate				ドロップアウト率.(0.0～1.0).(0.0＝ドロップアウトなし,1.0=全入力無視) */
		virtual Gravisbell::GUID AddDropoutLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_rate) = 0;

		/** ガウスノイズレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_average			発生する乱数の平均値
			@param	i_variance			発生する乱数の分散 */
		virtual Gravisbell::GUID AddGaussianNoiseLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_average, F32 i_variance) = 0;

		/** プーリングレイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_filterSize			プーリング幅.
			@param	i_stride				フィルタ移動量. */
		virtual Gravisbell::GUID AddPoolingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, Vector3D<S32> i_stride) = 0;

		/** バッチ正規化レイヤー.
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		virtual Gravisbell::GUID AddBatchNormalizationLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** バッチ正規化レイヤー(チャンネル区別なし)
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		virtual Gravisbell::GUID AddBatchNormalizationAllLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** スケール正規化レイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		virtual Gravisbell::GUID AddNormalizationScaleLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** 広域平均プーリングレイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		virtual Gravisbell::GUID AddGlobalAveragePoolingLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** GANにおけるDiscriminatorの出力レイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID. */
		virtual Gravisbell::GUID AddActivationDiscriminatorLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** アップサンプリングレイヤー
			@param	i_inputLayerGUID		追加レイヤーの入力先レイヤーのGUID.
			@param	i_upScale				拡張率.
			@param	i_paddingUseValue		拡張部分の穴埋めに隣接する値を使用するフラグ. (true=UpConvolution, false=TransposeConvolution) */
		virtual Gravisbell::GUID AddUpSamplingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_upScale, bool i_paddingUseValue) = 0;

		/** チャンネル抽出レイヤー. 入力されたレイヤーの特定チャンネルを抽出する. 入力/出力データ構造でX,Y,Zは同じサイズ.
			@param	i_inputLayerGUID	追加レイヤーの入力先レイヤーのGUID.
			@param	i_startChannelNo	開始チャンネル番号.
			@param	i_channelCount		抽出チャンネル数. */
		virtual Gravisbell::GUID AddChooseChannelLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_startChannelNo, U32 i_channelCount) = 0;

		/** XYZ抽出レイヤー. 入力されたレイヤーの特定XYZ区間を抽出する. 入力/出力データ構造でCHは同じサイズ.
			@param	startPosition	開始XYZ位置.
			@param	boxSize			抽出XYZ数. */
		virtual Gravisbell::GUID AddChooseBoxLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> startPosition, Vector3D<S32> boxSize) = 0;

		/** 出力データ構造変換レイヤー.
			@param	ch	CH数.
			@param	x	X軸.
			@param	y	Y軸.
			@param	z	Z軸. */
		virtual Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 ch, U32 x, U32 y, U32 z) = 0;
		/** 出力データ構造変換レイヤー.
			@param	outputDataStruct 出力データ構造 */
		virtual Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, const IODataStruct& outputDataStruct) = 0;

		/** X=0でミラー化する */
		virtual Gravisbell::GUID AddReshapeMirrorXLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;
		/** X=0で平方化する */
		virtual Gravisbell::GUID AddReshapeSquareCenterCrossLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;
		/** X=0で平方化する.
			入力信号数は1次元配列で(x-1)*(y-1)+1以上の要素数が必要.
			@param	x	X軸.
			@param	y	Y軸. */
		virtual Gravisbell::GUID AddReshapeSquareZeroSideLeftTopLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 x, U32 y) = 0;

		/** 信号の配列から値へ変換.
			@param	outputMinValue	出力の最小値
			@param	outputMaxValue	出力の最大値
			@param	resolution		分解能. 指定しない場合は現時点の出力チャンネル数が入る*/
		virtual Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue) = 0;
		virtual Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::U32 resolution) = 0;
		/** 確率の配列から値へ変換.
			@param	outputMinValue	出力の最小値
			@param	outputMaxValue	出力の最大値
			@param	resolution		分解能. 指定しない場合は現時点の出力チャンネル数が入る
			@param	variance		分散. 入力に対する教師信号を作成する際の正規分布の分散 */
		virtual Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance) = 0;
		virtual Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance, Gravisbell::U32 resolution) = 0;
		/** 値から信号の配列へ変換.
			@param	outputMinValue	出力の最小値
			@param	outputMaxValue	出力の最大値
			@param	resolution		分解能*/
		virtual Gravisbell::GUID AddValue2SignalArrayLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 inputMinValue, Gravisbell::F32 inputMaxValue, Gravisbell::U32 resolution) = 0;


	protected:
		//=============================
		// 入力結合レイヤー(チャンネル結合)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		virtual Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeInputLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeInputLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeInputLayer(std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeInputLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// 入力結合レイヤー(加算)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		virtual Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeAddLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeAddLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// 入力結合レイヤー(平均)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		virtual Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeAverageLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeAverageLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// 入力結合レイヤー(最大値)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		virtual Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** 入力結合レイヤー. 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeMaxLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** 入力結合レイヤー. 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeMaxLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}



	protected:
		//=============================
		// 入力結合レイヤー(乗算)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーの値を乗算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		virtual Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;
		
		/** 入力結合レイヤー. 入力されたレイヤーの値を乗算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeMultiplyLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}
		
		/** 入力結合レイヤー. 入力されたレイヤーの値を乗算する. 入力データ構造はX,Y,Zで同じサイズである必要がある. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeMultiplyLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// 入力結合レイヤー(加算)(旧式)
		//=============================
		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる. */
		virtual Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる. */
		Gravisbell::GUID AddResidualLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddResidualLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる. */
		template<typename... Rest>
		Gravisbell::GUID AddResidualLayer(std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddResidualLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	public:
		/** 入力結合レイヤー. 入力されたレイヤーのCHを結合する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeInputLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeAddLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeAverageLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** 入力結合レイヤー. 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeMaxLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}
		
		/** 入力結合レイヤー. 入力されたレイヤーの値を乗算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeMultiplyLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** 入力結合レイヤー. 入力されたレイヤーの値を合算する. 出力されるレイヤーのサイズは全サイズのうちの最大値になる.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddResidualLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	public:
		//==================================
		// 特殊レイヤー
		//==================================
		/** ニューラルネットワークにConvolution, Activationを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにConvolution, BatchNormalization, Activationを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにConvolution, BatchNormalization, Noise, Activationを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBNA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにBatchNormalization, Activation, Convolutionを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにBatchNormalization, Activation. Fully-Connectを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAF(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにFully-Connect, Activationを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FA(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにFully-Connect, Activation, Dropoutを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FAD(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにBatchNormalization, Noise, Activation, Convolutionを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BNAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにNoise, BatchNormalization, Activation, Convolutionを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_NBAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにConvolution, Activation, DropOutを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにConvolution, BatchNormalization, Activation, DropOutを行うレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにResNetレイヤーを追加する.(ノイズ付き) */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNet(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにResNetレイヤーを追加する. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_front_layerCount, U32 i_back_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ニューラルネットワークにResNetレイヤーを追加する.
			前半/後半に分けずに後半部分だけで処理. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize_single(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;
	};

	/** ニューラルネットワーク作成クラスを取得する */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	INeuralNetworkMaker* CreateNeuralNetworkManaker(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount);

}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif