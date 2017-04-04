//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
//======================================
#include<Layer/NeuralNetwork/INeuralNetwork.h>


#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>

#include"LayerConnect.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_Base : public INeuralNetwork
	{
	private:
		std::map<Gravisbell::GUID, ILayerConnect*>	lpLayerInfo;	/**< 全レイヤーの管理クラス. <レイヤーGUID, レイヤー接続情報のアドレス> */

		std::list<ILayerConnect*> lpCalculateLayer0List;		/**< レイヤーを処理順に並べたリスト.  */

		ILayerConnect* pOutputLayer;	/**< 出力信号に設定されているレイヤーのアドレス. */

	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_Base();
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_Base();

	public:
		//====================================
		// レイヤーの追加
		//====================================
		/** レイヤーを追加する.
			追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param pLayer	追加するレイヤーのアドレス. */
		ErrorCode AddLayer(INNLayer* pLayer);
		/** レイヤーを削除する.
			@param i_guid	削除するレイヤーのGUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** レイヤーを全削除する */
		ErrorCode EraseAllLayer();

		/** 登録されているレイヤー数を取得する */
		ErrorCode GetLayerCount()const;
		/** レイヤーを番号指定で取得する */
		const INNLayer* GetLayerByNum(const U32 i_layerNum);
		/** レイヤーをGUID指定で取得する */
		const INNLayer* GetLayerByGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// 入出力レイヤー
		//====================================
		/** 入力信号に割り当てられているGUIDを取得する */
		GUID GetInputGUID()const;

		/** 出力信号に割り当てらているレイヤーのGUIDを取得する */
		GUID GetOutputLayerGUID()const;
		/** 出力信号レイヤーを設定する */
		GUID SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// レイヤーの接続
		//====================================
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
		ErrorCode GetInputLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;

		/** レイヤーに接続しているバイパスレイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const;
		/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
			@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
		ErrorCode GetBypassLayerByNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const;


		/** レイヤーの接続状態に異常がないかチェックする.
			@param	o_errorLayer	エラーが発生したレイヤーGUID格納先. 
			@return	接続に異常がない場合はNO_ERROR, 異常があった場合は異常内容を返し、対象レイヤーのGUIDをo_errorLayerに格納する. */
		ErrorCode CheckAllConnection(Gravisbell::GUID& o_errorLayer);



		//====================================
		// 学習設定
		//====================================
		/** 学習設定を取得する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID. */
		const SettingData::Standard::IData GetLearnSettingData(const Gravisbell::GUID& guid);

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			@param	guid	取得対象レイヤーのGUID
			@param	data	設定する学習設定 */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& data);

		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型、enum型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			int型、float型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			bool型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param);
		/** 学習設定を設定する.
			設定した値はPreProcessLearnLoopを呼び出した際に適用される.
			string型が対象.
			@param	guid		取得対象レイヤーのGUID
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param);

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell