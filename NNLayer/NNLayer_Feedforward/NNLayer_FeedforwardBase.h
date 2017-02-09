//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include<INNLayer.h>

#include<vector>

using namespace CustomDeepNNLibrary;

typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

class NNLayer_FeedforwardBase : public CustomDeepNNLibrary::INNLayer
{
protected:
	GUID guid;

	INNLayerConfig* pConfig;

	std::vector<IOutputLayer*> lppInputFromLayer;		/**< ���͌����C���[�̃��X�g */
	std::vector<IInputLayer*>  lppOutputToLayer;	/**< �o�͐惌�C���[�̃��X�g */

public:
	/** �R���X�g���N�^ */
	NNLayer_FeedforwardBase(GUID guid);

	/** �f�X�g���N�^ */
	virtual ~NNLayer_FeedforwardBase();

	//===========================
	// ���C���[����
	//===========================
public:
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	unsigned int GetUseBufferByteCount()const;

	/** ���C���[�ŗL��GUID���擾���� */
	ELayerErrorCode GetGUID(GUID& o_guid)const;

	/** ���C���[�̎�ގ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	ELayerErrorCode GetLayerCode(GUID& o_layerCode)const;

	/** �ݒ����ݒ� */
	ELayerErrorCode SetLayerConfig(const INNLayerConfig& config);
	/** ���C���[�̐ݒ�����擾���� */
	const INNLayerConfig* GetLayerConfig()const;


	//===========================
	// ���̓��C���[�֘A
	//===========================
public:
	/** ���̓o�b�t�@�����擾����. */
	unsigned int GetInputBufferCount()const;

public:
	/** ���͌����C���[�ւ̃����N��ǉ�����.
		@param	pLayer	�ǉ�������͌����C���[
		@return	���������ꍇ0 */
	ELayerErrorCode AddInputFromLayer(IOutputLayer* pLayer);
	/** ���͌����C���[�ւ̃����N���폜����.
		@param	pLayer	�폜������͌����C���[
		@return	���������ꍇ0 */
	ELayerErrorCode EraseInputFromLayer(IOutputLayer* pLayer);

public:
	/** ���͌����C���[�����擾���� */
	unsigned int GetInputFromLayerCount()const;
	/** ���͌����C���[�̃A�h���X��ԍ��w��Ŏ擾����.
		@param num	�擾���郌�C���[�̔ԍ�.
		@return	���������ꍇ���͌����C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
	IOutputLayer* GetInputFromLayerByNum(unsigned int num)const;

	/** ���͌����C���[�����̓o�b�t�@�̂ǂ̈ʒu�ɋ��邩��Ԃ�.
		���Ώۓ��̓��C���[�̑O�ɂ����̓��̓o�b�t�@�����݂��邩.
		�@�w�K�����̎g�p�J�n�ʒu�Ƃ��Ă��g�p����.
		@return ���s�����ꍇ���̒l���Ԃ�*/
	int GetInputBufferPositionByLayer(const IOutputLayer* pLayer);


	//===========================
	// �o�̓��C���[�֘A
	//===========================
public:
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct GetOutputDataStruct()const;

	/** �o�̓o�b�t�@�����擾���� */
	unsigned int GetOutputBufferCount()const;

public:
	/** �o�͐惌�C���[�ւ̃����N��ǉ�����.
		@param	pLayer	�ǉ�����o�͐惌�C���[
		@return	���������ꍇ0 */
	ELayerErrorCode AddOutputToLayer(class IInputLayer* pLayer);
	/** �o�͐惌�C���[�ւ̃����N���폜����.
		@param	pLayer	�폜����o�͐惌�C���[
		@return	���������ꍇ0 */
	ELayerErrorCode EraseOutputToLayer(class IInputLayer* pLayer);

public:
	/** �o�͐惌�C���[�����擾���� */
	unsigned int GetOutputToLayerCount()const;
	/** �o�͐惌�C���[�̃A�h���X��ԍ��w��Ŏ擾����.
		@param num	�擾���郌�C���[�̔ԍ�.
		@return	���������ꍇ�o�͐惌�C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
	IInputLayer* GetOutputToLayerByNum(unsigned int num)const;


	//===========================
	// �ŗL�֐�
	//===========================
public:
	/** �j���[���������擾���� */
	unsigned int GetNeuronCount()const;
};
