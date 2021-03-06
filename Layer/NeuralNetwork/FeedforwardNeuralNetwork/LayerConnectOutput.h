//======================================
// レイヤー間の接続設定用クラス
//======================================
#ifndef __GRAVISBELL_LAYER_CONNECT_OUTPUT_H__
#define __GRAVISBELL_LAYER_CONNECT_OUTPUT_H__

#include<Layer/NeuralNetwork/INeuralNetwork.h>

#include"FeedforwardNeuralNetwork_FUNC.hpp"

#include<map>
#include<set>
#include<list>
#include<vector>

#include"LayerConnect.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** レイヤーの接続に関するクラス(出力信号の代用) */
	class LayerConnectOutput : public ILayerConnect
	{
	public:
		class FeedforwardNeuralNetwork_Base& neuralNetwork;

		std::vector<ILayerConnect*> lppInputFromLayer;	/**< 入力元レイヤー. SingleInput扱いなので、必ず1個 */

		S32 dInputBufferID;	/**< 入力誤差バッファID */

	public:
		/** コンストラクタ */
		LayerConnectOutput(class FeedforwardNeuralNetwork_Base& neuralNetwork);
		/** デストラクタ */
		virtual ~LayerConnectOutput();

	public:
		/** GUIDを取得する */
		Gravisbell::GUID GetGUID()const;
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKind()const;


		//====================================
		// 実行時設定
		//====================================
		/** 実行時設定を取得する. */
		const SettingData::Standard::IData* GetRuntimeParameter()const;

		/** 実行時設定を設定する.
			int型、float型、enum型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param);
		/** 実行時設定を設定する.
			int型、float型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param);
		/** 実行時設定を設定する.
			bool型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param);
		/** 実行時設定を設定する.
			string型が対象.
			@param	i_dataID	設定する値のID.
			@param	i_param		設定する値. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param);


		//====================================
		// 入出力データ構造
		//====================================
		/** 出力データ構造を取得する.
			@return	出力データ構造 */
		IODataStruct GetOutputDataStruct()const;
		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer_d()const;

		/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
		S32 GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const;
		/** 入力誤差バッファを位置指定で取得する */
		CONST_BATCH_BUFFER_POINTER GetDInputBufferByNum_d(S32 num)const;

		/** レイヤーリストを作成する.
			@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
		ErrorCode CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const;
		/** 計算順序リストを作成する.
			@param	i_lpLayerGUID		全レイヤーのGUID.
			@param	io_lpCalculateList	演算順に並べられた接続リスト.
			@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
			@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
		ErrorCode CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList);

	public:
		/** レイヤーに入力レイヤーを追加する. */
		ErrorCode AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
		/** レイヤーにバイパスレイヤーを追加する.*/
		ErrorCode AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

		/** レイヤーから入力レイヤーを削除する */
		ErrorCode EraseInputLayer(const Gravisbell::GUID& guid);
		/** レイヤーからバイパスレイヤーを削除する */
		ErrorCode EraseBypassLayer(const Gravisbell::GUID& guid);

		/** レイヤーの入力レイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetInputLayer();
		/** レイヤーのバイパスレイヤー設定をリセットする.
			@param	layerGUID	リセットするレイヤーのGUID. */
		ErrorCode ResetBypassLayer();

		/** レイヤーに接続している入力レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetInputLayerCount()const;
		/** レイヤーに接続している入力レイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetInputLayerByNum(U32 i_inputNum);

		/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
		U32 GetBypassLayerCount()const;
		/** レイヤーに接続しているバイパスレイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetBypassLayerByNum(U32 i_inputNum);

		/** レイヤーの接続を解除 */
		ErrorCode Disconnect(void);


		/** レイヤーで使用する出力バッファのIDを登録する */
		ErrorCode SetOutputBufferID(S32 i_outputBufferID);

		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ErrorCode SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID);
		/** レイヤーで使用する入力誤差バッファのIDを取得する
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		S32 GetDInputBufferID(U32 i_inputNum)const;


		//==========================================
		// 学習フラグ関連
		//==========================================
	public:
		/** 学習固定レイヤーフラグ.
			学習固定レイヤー(学習が必要ないレイヤー)の場合trueが返る. */
		bool IsFixLayer(void)const;

		/** 入力誤差の計算が必要なフラグ.
			必要な場合trueが返る. */
		bool IsNecessaryCalculateDInput(void)const;

		/** 誤差伝搬が必要なフラグ.
			誤差伝搬が必要な場合はtrueが返る.falseが返った場合、これ以降誤差伝搬を一切必要としない. */
		bool IsNecessaryBackPropagation(void)const;



		//==========================================
		// 出力レイヤー関連
		//==========================================
	public:
		/** 出力先レイヤーを追加する */
		ErrorCode AddOutputToLayer(ILayerConnect* pOutputToLayer);
		/** 出力先レイヤーを削除する */
		ErrorCode EraseOutputToLayer(const Gravisbell::GUID& guid);

		/** レイヤーに接続している出力先レイヤーの数を取得する.
			@param	i_layerGUID		接続されているレイヤーのGUID. */
		U32 GetOutputToLayerCount()const;
		/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
			@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
		ILayerConnect* GetOutputToLayerByNum(U32 i_num);


	public:
		/** レイヤーの初期化処理.
			接続状況は維持したままレイヤーの中身を初期化する. */
		ErrorCode Initialize(void);

		/** 接続の確立を行う */
		ErrorCode EstablishmentConnection(void);

		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(unsigned int batchSize);
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);


		/** 処理ループの初期化処理.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessLoop();


		/** 演算処理を実行する. */
		ErrorCode Calculate(void);
		/** 入力誤差計算を実行する. */
		ErrorCode CalculateDInput(void);
		/** 学習処理を実行する. */
		ErrorCode Training(void);

	};

		
}	// Gravisbell
}	// Layer
}	// NeuralNetwork


#endif	// __GRAVISBELL_LAYER_CONNECT_OUTPUT_H__