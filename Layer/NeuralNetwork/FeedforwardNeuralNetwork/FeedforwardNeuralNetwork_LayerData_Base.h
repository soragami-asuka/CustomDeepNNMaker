//======================================
// フィードフォワードニューラルネットワークの処理レイヤーのデータ
// 複数のレイヤーを内包し、処理する
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_LAYERDATA_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_LAYERDATA_BASE_H__

#include<set>
#include<map>

#include<Layer/NeuralNetwork/INNLayerConnectData.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_LayerData_Base : public INNLayerConnectData
	{
	protected:
		/** レイヤーデータ間の接続情報定義 */
		struct LayerConnect
		{
			Gravisbell::GUID guid;		/**< レイヤー自身のGUID */
			INNLayerData* pLayerData;	/**< レイヤーデータ本体 */
			std::set<Gravisbell::GUID> lpInputLayerGUID;	/**< 入力レイヤーのGUID一覧 */
			std::set<Gravisbell::GUID> lpBypassLayerGUID;	/**< バイパスレイヤーのGUID一覧 */

			LayerConnect()
				:	pLayerData	(NULL)
			{
			}
			LayerConnect(const Gravisbell::GUID guid, INNLayerData* pLayerData)
				:	guid		(guid)
				,	pLayerData	(pLayerData)
			{
			}
			LayerConnect(const LayerConnect& data)
				:	guid				(data.guid)
				,	pLayerData			(data.pLayerData)
				,	lpInputLayerGUID	(data.lpInputLayerGUID)
				,	lpBypassLayerGUID	(data.lpBypassLayerGUID)
			{
			}
			const LayerConnect& operator=(const LayerConnect& data)
			{
				this->guid = data.guid;
				this->pLayerData = data.pLayerData;
				this->lpInputLayerGUID = data.lpInputLayerGUID;
				this->lpBypassLayerGUID = data.lpBypassLayerGUID;

				return *this;
			}
		};

	protected:
		const ILayerDLLManager& layerDLLManager;

		const Gravisbell::GUID guid;			/**< レイヤー識別用のGUID */
		const Gravisbell::GUID inputLayerGUID;	/**< 入力信号に割り当てられているGUID.入力信号レイヤーの代用として使用する. */
		Gravisbell::GUID outputLayerGUID;		/**< 出力信号に割り当てられているGUID. */

		std::map<Gravisbell::GUID, INNLayerData*> lpLayerData;	/**< レイヤーデータGUID, レイヤーデータ */
		std::map<Gravisbell::GUID, LayerConnect> lpConnectInfo;	/**< レイヤーGUID, レイヤー接続情報 */

		IODataStruct inputDataStruct;	/**< 入力データ構造 */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */


		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_LayerData_Base(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_LayerData_Base();


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);
		/** 初期化. 各ニューロンの値をランダムに初期化
			@param	i_config			設定情報
			@oaram	i_inputDataStruct	入力データ構造情報
			@return	成功した場合0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize);


		//===========================
		// 共通制御
		//===========================
	public:
		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** レイヤーDLLマネージャの取得 */
		const ILayerDLLManager& GetLayerDLLManager(void)const;

		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U32 GetUseBufferByteCount()const;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** 設定情報を設定 */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		U32 GetOutputBufferCount()const;


	public:
		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		virtual INNLayer* CreateLayer(const Gravisbell::GUID& guid) = 0;

		/** 作成された新規ニューラルネットワークに対して内部レイヤーを追加する */
		ErrorCode AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork);


		//====================================
		// レイヤーの追加/削除/管理
		//====================================
	public:
		/** レイヤーデータを追加する.
			@param	i_guid			追加するレイヤーに割り当てられるGUID.
			@param	i_pLayerData	追加するレイヤーデータのアドレス. */
		ErrorCode AddLayer(const Gravisbell::GUID& i_guid, INNLayerData* i_pLayerData);
		/** レイヤーデータを削除する.
			@param i_guid	削除するレイヤーのGUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** レイヤーデータを全削除する */
		ErrorCode EraseAllLayer();

		/** 登録されているレイヤー数を取得する */
		U32 GetLayerCount();
		/** レイヤーのGUIDを番号指定で取得する */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);

		/** 登録されているレイヤーデータを番号指定で取得する */
		INNLayerData* GetLayerDataByNum(U32 i_layerNum);
		/** 登録されているレイヤーデータをGUID指定で取得する */
		INNLayerData* GetLayerDataByGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// 入出力レイヤー
		//====================================
	public:
		/** 入力信号に割り当てられているGUIDを取得する */
		GUID GetInputGUID();

		/** 出力信号レイヤーを設定する */
		ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// レイヤーの接続
		//====================================
	public:
		/** レイヤーに入力レイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** レイヤーにバイパスレイヤーを追加する.
			@param	receiveLayer	入力を受け取るレイヤー
			@param	postLayer		入力を渡す(出力する)レイヤー. */
		ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID);
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID);

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID);
		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID);
		/** レイヤーに接続している出力レイヤーの数を取得する */
		U32 GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID);

		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID);
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID);
		/** レイヤーに接続している出力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_outputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetOutputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_outputNum, Gravisbell::GUID& o_postLayerGUID);

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif