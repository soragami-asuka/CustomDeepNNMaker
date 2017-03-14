//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"Feedforward_DATA.hpp"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class FeedforwardBase : public Gravisbell::Layer::NeuralNetwork::INNLayer
	{
	protected:
		GUID guid;	/**< レイヤー識別用のGUID */

		IODataStruct inputDataStruct;	/**< 入力データ構造 */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		SettingData::Standard::IData* pLearnData;		/**< 学習設定を定義したコンフィグクラス */

		Feedforward::LayerStructure layerStructure;	/**< レイヤー構造 */
		Feedforward::LearnDataStructure learnData;	/**< 学習設定 */

		unsigned int batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		FeedforwardBase(GUID guid);

		/** デストラクタ */
		virtual ~FeedforwardBase();

		//===========================
		// レイヤー共通
		//===========================
	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		unsigned int GetLayerKindBase()const;

		/** レイヤー固有のGUIDを取得する */
		ErrorCode GetGUID(GUID& o_guid)const;

		/** レイヤーの種類識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		ErrorCode GetLayerCode(GUID& o_layerCode)const;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		unsigned int GetBatchSize()const;


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** 設定情報を設定 */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);
		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerConfig()const;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		unsigned int GetUseBufferByteCount()const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		unsigned int GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		unsigned int GetOutputBufferCount()const;


		//===========================
		// 固有関数
		//===========================
	public:
		/** ニューロン数を取得する */
		unsigned int GetNeuronCount()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell