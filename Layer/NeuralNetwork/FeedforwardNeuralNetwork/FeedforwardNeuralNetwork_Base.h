//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_BASE_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>

#include"LayerConnect.h"
#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_Base : public INeuralNetwork
	{
	private:
		// データ本体
		class FeedforwardNeuralNetwork_LayerData_Base& layerData;

		std::map<Gravisbell::GUID, ILayerConnect*>	lpLayerInfo;	/**< 全レイヤーの管理クラス. <レイヤーGUID, レイヤー接続情報のアドレス> */
		std::vector<ILayerData*>	lpTemporaryLayerData;			/**< 一時保存されたレイヤーデータ. */

		std::list<ILayerConnect*> lpCalculateLayerList;		/**< レイヤーを処理順に並べたリスト.  */

		const Gravisbell::GUID guid;			/**< レイヤー識別用のGUID */
		IODataStruct outputDataStruct;	/**< 出力データ構造 */

		SettingData::Standard::IData* pLearnData;		/**< 学習設定を定義したコンフィグクラス */

		U32 batchSize;	/**< バッチサイズ */

	protected:
		std::vector<LayerConnectInput*> lppInputLayer;	/**< 入力信号の代替レイヤーのアドレス. */
		LayerConnectOutput outputLayer;	/**< 出力信号の代替レイヤーのアドレス. */

		Gravisbell::Common::ITemporaryMemoryManager* pLocalTemporaryMemoryManager;
		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

		// 入出力バッファ
		//std::vector<F32>		lpInputBuffer;		/**< 入力バッファ <バッチ数><入力信号数> */
		std::vector<std::vector<F32>>		lppInputTmpBuffer;			/**< 入力バッファ本体 <インプットレイヤー数><バッチ数*入力信号数> */
		std::vector<const F32*>					lppInputBuffer;				/**< 入力バッファのアドレス <インプットレイヤー数> */

		CONST_BATCH_BUFFER_POINTER*	m_lppInputBuffer;	/**< 外部から預かった入力バッファのアドレス(演算デバイス依存) */
		BATCH_BUFFER_POINTER*		m_lppDInputBuffer;	/**< 外部から預かった入力誤差バッファのアドレス(演算デバイス依存) */
		BATCH_BUFFER_POINTER		m_lppDOutputBuffer;	/**< 外部から預かった出力誤差バッファのアドレス(演算デバイス依存) */

	public:
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
		/** コンストラクタ */
		FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager* i_pTemporaryMemoryManager);
		/** コンストラクタ */
		FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_Base();


		//====================================
		// レイヤーの追加
		//====================================
	public:
		/** レイヤーを追加する.
			追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param	pLayer				追加するレイヤーのアドレス.
			@param	i_onLayerFixFlag	レイヤー固定化フラグ. */
		ErrorCode AddLayer(ILayerBase* pLayer, bool i_onLayerFixFLag);

		/** 一時レイヤーを追加する.
			追加したレイヤーデータの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param	i_pLayerData	追加するレイヤーデータ.
			@param	o_player		追加されたレイヤーのアドレス. */
		virtual ErrorCode AddTemporaryLayer(ILayerData* i_pLayerData, ILayerBase** o_pLayer, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, bool i_onLyaerFixFlag);

		/** レイヤーを削除する.
			@param i_guid	削除するレイヤーのGUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** レイヤーを全削除する */
		ErrorCode EraseAllLayer();

		/** 登録されているレイヤー数を取得する */
		U32 GetLayerCount()const;
		/** レイヤーのGUIDを番号指定で取得する */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);


		//====================================
		// 入出力レイヤー
		//====================================
	public:
		/** 入力信号レイヤー数を取得する */
		U32 GetInputCount();
		/** 入力信号に割り当てられているGUIDを取得する */
		Gravisbell::GUID GetInputGUID(U32 i_inputLayerNum);

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
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;


		/** レイヤーの接続状態に異常がないかチェックする.
			@param	o_errorLayer	エラーが発生したレイヤーGUID格納先. 
			@return	接続に異常がない場合はNO_ERROR, 異常があった場合は異常内容を返し、対象レイヤーのGUIDをo_errorLayerに格納する. */
		ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer);

		//====================================
		// 入力誤差バッファ関連
		//====================================
	private:
		/** 各レイヤーが使用する入力誤差バッファを割り当てる */
		ErrorCode AllocateDInputBuffer(void);

	protected:
		/** 入力誤差バッファの総数を設定する */
		virtual ErrorCode SetDInputBufferCount(U32 i_DInputBufferCount) = 0;

		/** 入力誤差バッファのサイズを設定する */
		virtual ErrorCode ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** 途中計算用の入力誤差バッファを取得する(処理デバイス依存) */
		virtual BATCH_BUFFER_POINTER GetTmpDInputBuffer_d(U32 i_DInputBufferNo) = 0;
		/** 途中計算用の入力誤差バッファを取得する(処理デバイス依存) */
		virtual CONST_BATCH_BUFFER_POINTER GetTmpDInputBuffer_d(U32 i_DInputBufferNo)const = 0;


		//====================================
		// 出力バッファ関連
		//====================================
	private:
		/** 各レイヤーが使用する出力バッファを割り当てる */
		ErrorCode AllocateOutputBuffer(void);

	protected:
		/** 出力バッファの総数を設定する */
		virtual ErrorCode SetOutputBufferCount(U32 i_outputBufferCount) = 0;

		/** 出力バッファのサイズを設定する */
		virtual ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** 出力バッファの現在の使用者を取得する */
		virtual GUID GetReservedOutputBufferID(U32 i_i_outputBufferNo) = 0;
		/** 出力バッファを使用中にして取得する(処理デバイス依存) */
		virtual BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid) = 0;


		//====================================
		// 外部から預かった入出力バッファ関連
		//====================================
	public:
//		/** 入力バッファを取得する(処理デバイス依存) */
//		CONST_BATCH_BUFFER_POINTER GetInputBuffer();
		/** 入力バッファを取得する(処理デバイス依存) */
		CONST_BATCH_BUFFER_POINTER GetInputBuffer_d(U32 i_inputLayerNum);
		/** 入力誤差バッファを取得する(処理デバイス依存) */
		BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_inputLayerNum);
		/** 出力誤差バッファを取得する(処理デバイス依存) */
		BATCH_BUFFER_POINTER GetDOutputBuffer_d();

		/** NNが入力誤差バッファを保持しているかを確認する */
		bool CheckIsHaveDInputBuffer()const;

		//====================================
		// 学習設定
		//====================================
	public:
		/** 実行時設定を取得する. */
		const SettingData::Standard::IData* GetRuntimeParameter()const;
		SettingData::Standard::IData* GetRuntimeParameter();

		/** 学習設定を取得する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID. */
		const SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid)const;
		SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid);

		/** 学習設定のアイテムを取得する.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID. */
		SettingData::Standard::IItemBase* GetRuntimeParameterItem(const Gravisbell::GUID& guid, const wchar_t* i_dataID);

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param);
		ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param);

		/** レイヤーに学習禁止を設定する.
			@param	guid		設定対象レイヤーのGUID.
			@param	i_fixFlag	固定化フラグ.true=学習しない. */
		ErrorCode SetLayerFixFlag(const Gravisbell::GUID& guid, bool i_fixFlag);


		//====================================
		// 入出力バッファ関連
		//====================================
	public:
		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;


		//===========================
		// レイヤー共通
		//===========================
	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const;

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const override;

		/** レイヤーの種類識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const override;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		U32 GetBatchSize()const override;

		/** 一時バッファ管理クラスを取得する */
		Common::ITemporaryMemoryManager& GetTemporaryMemoryManager();

		//================================
		// 初期化処理
		//================================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データの数を取得する */
		U32 GetInputDataCount()const override;

		/** 入力レイヤー番号をIDから取得する.
			@return	入力レイヤーではない場合は-1,入力レイヤーである場合は番号を0以上で返す */
		S32 GetInputLayerNoByGUID(const Gravisbell::GUID& i_guid)const;

		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const override;

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount(U32 i_dataNum)const override;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const override;
		IODataStruct GetOutputDataStruct(const GUID& i_layerGUID)const;

		/** 出力バッファ数を取得する */
		U32 GetOutputBufferCount()const override;


		//================================
		// 演算処理
		//================================
	protected:
		/** 接続の確立を行う */
		ErrorCode EstablishmentConnection(void);

	public:
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(U32 batchSize)override;
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(unsigned int batchSize)override;
		
		/** ループの初期化処理.データセットの実行開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessLoop()override;


		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)override;


		//================================
		// 学習処理
		//================================
	public:
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)override;

		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER )override;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif	// __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_H__