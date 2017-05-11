//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[�̃f�[�^
// �����̃��C���[�����A��������
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
		/** ���C���[�f�[�^�Ԃ̐ڑ�����` */
		struct LayerConnect
		{
			Gravisbell::GUID guid;		/**< ���C���[���g��GUID */
			INNLayerData* pLayerData;	/**< ���C���[�f�[�^�{�� */
			std::set<Gravisbell::GUID> lpInputLayerGUID;	/**< ���̓��C���[��GUID�ꗗ */
			std::set<Gravisbell::GUID> lpBypassLayerGUID;	/**< �o�C�p�X���C���[��GUID�ꗗ */

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

		const Gravisbell::GUID guid;			/**< ���C���[���ʗp��GUID */
		const Gravisbell::GUID inputLayerGUID;	/**< ���͐M���Ɋ��蓖�Ă��Ă���GUID.���͐M�����C���[�̑�p�Ƃ��Ďg�p����. */
		Gravisbell::GUID outputLayerGUID;		/**< �o�͐M���Ɋ��蓖�Ă��Ă���GUID. */

		std::map<Gravisbell::GUID, INNLayerData*> lpLayerData;	/**< ���C���[�f�[�^GUID, ���C���[�f�[�^ */
		std::map<Gravisbell::GUID, LayerConnect> lpConnectInfo;	/**< ���C���[GUID, ���C���[�ڑ���� */

		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */


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
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize);


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
		U32 GetUseBufferByteCount()const;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		U32 GetOutputBufferCount()const;


	public:
		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		virtual INNLayer* CreateLayer(const Gravisbell::GUID& guid) = 0;

		/** �쐬���ꂽ�V�K�j���[�����l�b�g���[�N�ɑ΂��ē������C���[��ǉ����� */
		ErrorCode AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork);


		//====================================
		// ���C���[�̒ǉ�/�폜/�Ǘ�
		//====================================
	public:
		/** ���C���[�f�[�^��ǉ�����.
			@param	i_guid			�ǉ����郌�C���[�Ɋ��蓖�Ă���GUID.
			@param	i_pLayerData	�ǉ����郌�C���[�f�[�^�̃A�h���X. */
		ErrorCode AddLayer(const Gravisbell::GUID& i_guid, INNLayerData* i_pLayerData);
		/** ���C���[�f�[�^���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		ErrorCode EraseLayer(const Gravisbell::GUID& i_guid);
		/** ���C���[�f�[�^��S�폜���� */
		ErrorCode EraseAllLayer();

		/** �o�^����Ă��郌�C���[�����擾���� */
		U32 GetLayerCount();
		/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
		ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid);

		/** �o�^����Ă��郌�C���[�f�[�^��ԍ��w��Ŏ擾���� */
		INNLayerData* GetLayerDataByNum(U32 i_layerNum);
		/** �o�^����Ă��郌�C���[�f�[�^��GUID�w��Ŏ擾���� */
		INNLayerData* GetLayerDataByGUID(const Gravisbell::GUID& i_guid);


		//====================================
		// ���o�̓��C���[
		//====================================
	public:
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		GUID GetInputGUID();

		/** �o�͐M�����C���[��ݒ肷�� */
		ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid);


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

	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif