//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[�̃f�[�^
// �����̃��C���[�����A��������
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_LAYERDATA_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_LAYERDATA_BASE_H__

#include<set>
#include<map>
#include<vector>

#include<Layer/Connect/ILayerConnectData.h>
#include<Layer/ILayerData.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>
#include"FeedforwardNeuralNetwork_DATA.hpp"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	class FeedforwardNeuralNetwork_LayerData_Base : public Connect::ILayerConnectData
	{
	protected:
		/** ���C���[�f�[�^�Ԃ̐ڑ�����` */
		struct LayerConnect
		{
			Gravisbell::GUID guid;		/**< ���C���[���g��GUID */
			ILayerData* pLayerData;	/**< ���C���[�f�[�^�{�� */
			std::vector<Gravisbell::GUID> lpInputLayerGUID;	/**< ���̓��C���[��GUID�ꗗ */
			std::vector<Gravisbell::GUID> lpBypassLayerGUID;	/**< �o�C�p�X���C���[��GUID�ꗗ */
			bool onFixFlag;	/**< �Œ背�C���[�t���O */

			LayerConnect()
				:	pLayerData	(NULL)
			{
			}
			LayerConnect(const Gravisbell::GUID guid, ILayerData* pLayerData, bool onFixFlag)
				:	guid		(guid)
				,	pLayerData	(pLayerData)
				,	onFixFlag	(onFixFlag)
			{
			}
			LayerConnect(const LayerConnect& data)
				:	guid				(data.guid)
				,	pLayerData			(data.pLayerData)
				,	lpInputLayerGUID	(data.lpInputLayerGUID)
				,	lpBypassLayerGUID	(data.lpBypassLayerGUID)
				,	onFixFlag			(data.onFixFlag)
			{
			}
			const LayerConnect& operator=(const LayerConnect& data)
			{
				this->guid = data.guid;
				this->pLayerData = data.pLayerData;
				this->lpInputLayerGUID = data.lpInputLayerGUID;
				this->lpBypassLayerGUID = data.lpBypassLayerGUID;
				this->onFixFlag = data.onFixFlag;

				return *this;
			}
		};

	protected:
		const ILayerDLLManager& layerDLLManager;

		const Gravisbell::GUID guid;			/**< ���C���[���ʗp��GUID */
		std::vector<Gravisbell::GUID> lpInputLayerGUID;	/**< ���͐M���Ɋ��蓖�Ă��Ă���GUID.���͐M�����C���[�̑�p�Ƃ��Ďg�p����. */
		Gravisbell::GUID outputLayerGUID;		/**< �o�͐M���Ɋ��蓖�Ă��Ă���GUID. */

		std::map<Gravisbell::GUID, ILayerData*> lpLayerData;	/**< ���C���[�f�[�^GUID, ���C���[�f�[�^ */
		std::map<Gravisbell::GUID, LayerConnect> lpConnectInfo;	/**< ���C���[GUID, ���C���[�ڑ���� */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		FeedforwardNeuralNetwork::LayerStructure layerStructure;	/**< ���C���[�\�� */


		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_LayerData_Base(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_LayerData_Base();


		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@param	i_config			�ݒ���
			@oaram	i_inputDataStruct	���̓f�[�^�\�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize);


		//===========================
		// ���ʐ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** ���C���[DLL�}�l�[�W���̎擾 */
		const ILayerDLLManager& GetLayerDLLManager(void)const;

		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U64 GetUseBufferByteCount()const;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;



	public:
		//===========================
		// ���C���[�\��
		//===========================
		/** ���̓f�[�^�\�����g�p�\���m�F����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
		bool CheckCanUseInputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
		bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);


		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		IODataStruct GetOutputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
		IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);


		/** �����o�͂��\�����m�F���� */
		bool CheckCanHaveMultOutputLayer(void);

	private:
		const IODataStruct* tmp_lpInputDataStruct;
		U32 tmp_inputLayerCount;
		std::map<Gravisbell::GUID, IODataStruct> tmp_lpOutputDataStruct;

		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		IODataStruct GetOutputDataStruct(const Gravisbell::GUID& i_guid);

	public:
		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;

	protected:
		/** �쐬���ꂽ�V�K�j���[�����l�b�g���[�N�ɑ΂��ē������C���[��ǉ����� */
		ErrorCode AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);


		//====================================
		// ���C���[�̒ǉ�/�폜/�Ǘ�
		//====================================
	public:
		/** ���C���[�f�[�^��ǉ�����.
			@param	i_guid			�ǉ����郌�C���[�Ɋ��蓖�Ă���GUID.
			@param	i_pLayerData	�ǉ����郌�C���[�f�[�^�̃A�h���X.
			@param	i_onFixFlag		���C���[���Œ艻����t���O. */
		ErrorCode AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData, bool i_onFixFlag);
		/** ���C���[�f�[�^���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** ���C���[�f�[�^��S�폜���� */
		ErrorCode EraseAllLayer();

		/** �o�^����Ă��郌�C���[�����擾���� */
		U32 GetLayerCount();
		/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);

		/** �o�^����Ă��郌�C���[��ԍ��w��Ŏ擾���� */
		LayerConnect* GetLayerByNum(U32 i_layerNum);
		/** �o�^����Ă��郌�C���[��GUID�w��Ŏ擾���� */
		LayerConnect* GetLayerByGUID(const Gravisbell::GUID& i_guid);
		LayerConnect* GetLayerByGUID(const Gravisbell::GUID& i_guid)const;

		/** �o�^����Ă��郌�C���[�f�[�^��ԍ��w��Ŏ擾���� */
		ILayerData* GetLayerDataByNum(U32 i_layerNum);
		/** �o�^����Ă��郌�C���[�f�[�^��GUID�w��Ŏ擾���� */
		ILayerData* GetLayerDataByGUID(const Gravisbell::GUID& i_guid);

		/** ���C���[�̌Œ艻�t���O���擾���� */
		bool GetLayerFixFlagByGUID(const Gravisbell::GUID& i_guid);

		//====================================
		// ���o�̓��C���[
		//====================================
	public:
		/** ���͐M�����C���[�����擾���� */
		U32 GetInputCount();
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		Gravisbell::GUID GetInputGUID(U32 i_inputLayerNum);

		/** �o�͐M�����C���[��ݒ肷�� */
		ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid);
		/** �o�͐M�����C���[��GUID���擾���� */
		Gravisbell::GUID GetOutputLayerGUID();


		//====================================
		// ���C���[�̐ڑ�
		//====================================
	public:
		/** ���C���[�ɓ��̓��C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** ���C���[������̓��C���[���폜����. 
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode EraseInputLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);
		/** ���C���[����o�C�p�X���C���[���폜����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		ErrorCode EraseBypassLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer);

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID);
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID);

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID);
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID);
		/** ���C���[�ɐڑ����Ă���o�̓��C���[�̐����擾���� */
		U32 GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID);

		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID);
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID);
		/** ���C���[�ɐڑ����Ă���o�̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_outputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		ErrorCode GetOutputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_outputNum, Gravisbell::GUID& o_postLayerGUID);


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]);
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif